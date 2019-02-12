import numpy as np
import tensorflow as tf
import pandas as pd
import cv2
import time
import os
import sys
import logging
import pprint
from sklearn.metrics import f1_score

def printProgress (iteration, total, prefix = '', suffix = '', decimals = 1, barLength = 100):
    formatStr = "{0:." + str(decimals) + "f}" 
    percent = formatStr.format(100 * (iteration / float(total))) 
    filledLength = int(round(barLength * iteration / float(total))) 
    bar = '#' * filledLength + '-' * (barLength - filledLength) 
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percent, '%', suffix))
    if (iteration + 1) == total:
        bar = '#' * barLength
        sys.stdout.write('\r%s |%s| %s %s' % (prefix, bar, '100%', suffix))
        sys.stdout.write('\n') 
    sys.stdout.flush()

def make_label_one_hot(df, num_classes):
    label_array = np.array(df.Target)
    labels_one_hot = np.zeros(shape=[len(label_array), num_classes])
    for i, lb in enumerate(label_array):
        for j in lb.split(' '):
            labels_one_hot[i][int(j)] = 1
    return labels_one_hot
    
def make_imgs(paths, want_size=512):
    imgs = []
    for filename in paths:
        img = [cv2.imread(filename+'_'+c+'.png', cv2.IMREAD_GRAYSCALE) for c in ['red', 'green', 'blue', 'yellow']]
        if want_size != img[0].shape[0]:
            img = [cv2.resize(im, (want_size, want_size), cv2.INTER_LINEAR) for im in img]
        img = np.concatenate([im[:, :, np.newaxis] for im in img], axis=-1)
        imgs.append(img)
    
    return np.array(imgs, dtype=np.uint8)

def make_df0(label):
    is_df0 = np.argmax(label, axis=1)
    df0 = np.array([1, 0], dtype=np.float32)
    df_ow = np.array([0, 1], dtype=np.float32)
    label_binary = []
    for i in range(len(label)):
        if is_df0[i] == 0:
            label_binary.append(df0)
        else:
            label_binary.append(df_ow)
    return np.array(label_binary, dtype=np.float32)    

def make_df_otherwise(label, specific):
    is_df0 = (label[:, specific] == 1).astype(np.int32)
    df = np.array([1, 0], dtype=np.float32)
    df_ow = np.array([0, 1], dtype=np.float32)
    label_binary = []
    for i in range(len(label)):
        if is_df0[i] == 1:
            label_binary.append(df)
        else:
            label_binary.append(df_ow)
    return np.array(label_binary, dtype=np.float32)    
    
def get_f1_score(paths, label, X, pred, sess, batch_size, want_size, num_classes, th, is_progress=False):
    """
    Inputs:
        paths :
        label : np.array or [].
        is_progress : if True, visualize progress.
    Returns:
        f1 score, predictions if label is given
        predictions if label is [], an empty list
    """
    total_batch = np.ceil(len(paths) / float(batch_size)).astype(np.int32)
    idxs = np.arange(len(paths))
    predictions = np.array([], dtype=np.int32).reshape((0, num_classes))
    for i in range(total_batch):
        idxs_i = idxs[i * batch_size : (i + 1)*batch_size]
        imgs = make_imgs(paths[idxs_i], want_size)
        preds = sess.run(pred, feed_dict={X : imgs})
        predictions = np.concatenate((predictions, preds), axis=0)
        if is_progress:
            printProgress(iteration=i, total=total_batch)
    if len(label) > 0:
        label = label.astype(np.int32)
        return f1_score(y_true=label, y_pred=(predictions > th).astype(np.int32), average='macro'), predictions
    else:
        return predictions

def get_accuracy(paths, label, X, logits, sess, batch_size, want_size):
    """
    Returns :
        accuracy, predictions
    """
    label = np.argmax(label, axis=-1)
    total_batch = np.ceil(len(paths) / float(batch_size)).astype(np.int32)
    idxs = np.arange(len(paths))
    predictions = np.array([], dtype=np.int32).reshape((0))
    for i in range(total_batch):
        idxs_i = idxs[i * batch_size : (i + 1)*batch_size]
        imgs = make_imgs(paths[idxs_i], want_size)
        pred = sess.run(logits, feed_dict={X : imgs})
        pred = np.argmax(pred, axis=1)
        predictions = np.concatenate((predictions, pred), axis=0)
    correct = (label == predictions)
    
    return correct.mean(), predictions
    
def make_pred_lists(predictions, forced=False):
    pred_lists = []
    for pred in predictions:
        string = ''
        flag = False
        for i, y in reversed(list(enumerate(pred))):
            if y == 1:
                string += '{} '.format(i)
                flag = True
        if forced:
            if not flag:
                string = '0 '
        string = string[:-1]
        pred_lists.append(string)
    return pred_lists
    
def batch_norm(inputs, name, momentum=0.9, epsilon=1e-5, is_training=True):
    return tf.contrib.layers.batch_norm(inputs=inputs, decay=momentum, 
                                        epsilon=epsilon, scale=True, 
                                        is_training=is_training, 
                                        updates_collections=None, scope=name+'_bn')

def Conv(inputs, filters, kernel_size=3, strides=(1, 1), padding='same', name='Conv'):
    return tf.layers.conv2d(inputs, filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, name=name)

def Conv_bn_act_repeat(inputs, repeats, filters, kernel_size=3, strides=(1, 1), padding='same', is_training=True, name='Conv_bn_act'):
    output = inputs
    for i in range(repeats):
        conv = Conv(output, filters, kernel_size, strides, padding, name+'_conv_{}'.format(i+1))
        bn = batch_norm(conv, name=name+'_bn_{}'.format(i+1), is_training=is_training)
        act = tf.nn.relu(bn, name=name+'_act_{}'.format(i+1))
        output = act
    return output
def Maxpolling(inputs, pool_size=2, strides=2, padding='same'):
    return tf.layers.max_pooling2d(inputs, pool_size=pool_size, strides=strides, padding=padding)      

def vgg16(inputs, num_classes, is_training):
    net = Conv_bn_act_repeat(inputs, repeats=2, filters=64, is_training=is_training, name='block1')
    net = Maxpolling(net)
    net = Conv_bn_act_repeat(net, repeats=2, filters=128, is_training=is_training, name='block2')
    net = Maxpolling(net)
    net = Conv_bn_act_repeat(net, repeats=3, filters=256, is_training=is_training, name='block3')
    net = Maxpolling(net)
    net = Conv_bn_act_repeat(net, repeats=3, filters=512, is_training=is_training, name='block4')
    net = Maxpolling(net)
    net = Conv_bn_act_repeat(net, repeats=3, filters=512, is_training=is_training, name='block5')
    
    net = tf.reduce_mean(net, axis=[1, 2], name='global_pool')
    
    net = tf.layers.flatten(net)
    net = tf.layers.dropout(net, rate=0.5, training=is_training, name='dropout')
    net = tf.layers.dense(net, units=num_classes, name='dense')
    
    return net

def vgg19(inputs, num_classes, is_training):
    net = Conv_bn_act_repeat(inputs, repeats=2, filters=64, is_training=is_training, name='block1')
    net = Maxpolling(net)
    net = Conv_bn_act_repeat(net, repeats=2, filters=128, is_training=is_training, name='block2')
    net = Maxpolling(net)
    net = Conv_bn_act_repeat(net, repeats=4, filters=256, is_training=is_training, name='block3')
    net = Maxpolling(net)
    net = Conv_bn_act_repeat(net, repeats=4, filters=512, is_training=is_training, name='block4')
    net = Maxpolling(net)
    net = Conv_bn_act_repeat(net, repeats=4, filters=512, is_training=is_training, name='block5')
    
    net = tf.reduce_mean(net, axis=[1, 2], name='global_pool')
    
    net = tf.layers.flatten(net)
    net = tf.layers.dropout(net, rate=0.5, training=is_training, name='dropout')
    net = tf.layers.dense(net, units=num_classes, name='dense')
    
    return net

def df_one(t, choose=0):
    df = tf.expand_dims(tf.fill(tf.shape(t[:, 0]), 1.0), axis=-1)
    df_ow = tf.expand_dims(tf.fill(tf.shape(t[:, 0]), 0.0), axis=-1)
    
    if choose == 0:
        return tf.concat((df, df_ow), axis=-1)
    else:
        return tf.concat((df_ow, df), axis=-1)    