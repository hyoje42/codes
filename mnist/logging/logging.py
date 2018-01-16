import tensorflow as tf
from keras.datasets import mnist
(img_train, label_train), (img_test, label_test) = mnist.load_data()

import numpy as np
import matplotlib.pyplot as plt
import time






model = 'oversample'

import logging

#logging.basicConfig(level=logging.DEBUG, filename='./test.log')

#logging.debug("debugging log")

logger = logging.getLogger('mylogger')
fileHandler = logging.FileHandler('./myLoggerTest_{}.log'.format(model))
streamHandler = logging.StreamHandler()

logger.addHandler(fileHandler)
logger.addHandler(streamHandler)

logger.setLevel(logging.DEBUG)









# split test data
test_img = {}
for idx in range(10):
    test_img[idx] = []

for idx in range(len(img_test)):
    test_img[label_test[idx]].append(img_test[idx])
for num in range(10):
    test_img[num] = np.array(test_img[num])

for num in range(10):
    logger.info("shape of {} : {}".format(num, test_img[num].shape))

# split training data
org_image = {}
for idx in range(10):
    org_image[idx] = []

for idx in range(len(img_train)):
    org_image[label_train[idx]].append(img_train[idx])
for num in range(10):
    org_image[num] = np.array(org_image[num])

for num in range(10):
    logger.info("shape of {} : {}".format(num, org_image[num].shape))

image = {}
helper = [1000, 1000, 300, 100, 100, 10, 10, 10, 10, 300]

for num in range(10):
    idxs = np.random.permutation(range(len(org_image[num])))
    image[num] = org_image[num][idxs[0:helper[num]]]
    logger.info("The number of {} : {}".format(num, image[num].shape))

from keras.utils import np_utils

img_row, img_col = img_train.shape[1], img_train.shape[2]
img_depth = 1

img_train = img_train.reshape(img_train.shape[0], img_row, img_col, img_depth)
img_test = img_test.reshape(img_test.shape[0], img_row, img_col, img_depth)
img_train = img_train.astype('float32')
img_test = img_test.astype('float32')
img_train /= 255
img_test /= 255

label_train = np_utils.to_categorical(label_train, 10)
label_test = np_utils.to_categorical(label_test, 10)

W = np.load('../Weights/weights_transfer_Conv1_Fc1_minor2_epcoh1.npy')

X = tf.placeholder(tf.float32, [None, img_row, img_col, img_depth])
Y = tf.placeholder(tf.float32, [None, 10])
training = tf.placeholder(tf.bool)

with tf.variable_scope('conv_layer'):
    conv1 = tf.layers.conv2d(inputs=X, filters=32, kernel_size=[3, 3], 
                             kernel_initializer=tf.constant_initializer(W[0]), bias_initializer=tf.constant_initializer(W[1]),
                            padding='SAME', activation=tf.nn.relu, name='conv1')
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], 
                                    padding='SAME', strides=2, name='pool1')
    conv2 = tf.layers.conv2d(inputs=pool1, filters=32, kernel_size=[3, 3], 
                             kernel_initializer=tf.constant_initializer(W[2]), bias_initializer=tf.constant_initializer(W[3]),
                            padding='SAME', activation=tf.nn.relu, name='conv2')
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], 
                                    padding='SAME', strides=2, name='pool2')
    conv3 = tf.layers.conv2d(inputs=pool2, filters=32, kernel_size=[3, 3],
                             kernel_initializer=tf.constant_initializer(W[4]), bias_initializer=tf.constant_initializer(W[5]),
                            padding='SAME', activation=tf.nn.relu, name='conv3')
    pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], 
                                padding='SAME', strides=2, name='pool3')
with tf.variable_scope('fc_layer'):
    flat = tf.reshape(pool3, [-1, 4*4*32], name='flat')
    dense1 = tf.layers.dense(inputs=flat, units=1024, activation=tf.nn.relu, 
                             kernel_initializer=tf.constant_initializer(W[6]), bias_initializer=tf.constant_initializer(W[7]),
                             name='dense1')
    drop1 = tf.layers.dropout(inputs=dense1, rate=0.5, training=training)
    dense2 = tf.layers.dense(inputs=drop1, units=512, activation=tf.nn.relu, 
                             kernel_initializer=tf.constant_initializer(W[8]), bias_initializer=tf.constant_initializer(W[9]),
                             name='dense2')
    drop2 = tf.layers.dropout(inputs=dense2, rate=0.5, training=training)
    logits = tf.layers.dense(inputs=drop2, units=10, 
                            kernel_initializer=tf.constant_initializer(W[10]), bias_initializer=tf.constant_initializer(W[11]),)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

def get_accuracy(sess, logits, img, label):
    if len(img) > 5000:
        num = int(len(img)/2)
        logit1 = sess.run(logits, feed_dict={X: img[0:num], Y: label[0:num], training: False})
        logit2 = sess.run(logits, feed_dict={X: img[num:num*2], Y: label[num:num*2], training: False})
        logit = np.concatenate((logit1, logit2), axis=0)
    else:
        num = len(img)
        logit = sess.run(logits, feed_dict={X: img, Y: label, training: False})
    
    equal = np.equal(np.argmax(logit, 1), np.argmax(label, 1))
    predict = equal.astype(np.float32)
    accuracy = np.mean(predict)
    
    return accuracy, predict, logit

sess = tf.Session()
sess.run(tf.global_variables_initializer())
test_accuracy, _, _ = get_accuracy(sess, logits, img_test, label_test)
logger.info ('Test Accuracy : {:.5f}'.format(test_accuracy))

for num in range(10):
    specific_label = np_utils.to_categorical(np.full((len(test_img[num])), num), 10)
    specific_test_accuracy, test_predict, test_logit = get_accuracy(sess, logits, np.expand_dims(test_img[num], axis=-1), 
                                                                    specific_label)
    logger.info("The specific accurcy of {} : {:.5f}".format(num, specific_test_accuracy))
acc, _, _ = get_accuracy(sess, logits, img_test, label_test)
logger.info("Total accuracy : {:.5f}".format(acc))

num = 5
specific_label = np_utils.to_categorical(np.full((len(test_img[num])), num), 10)
specific_test_accuracy, test_predict, test_logit = get_accuracy(sess, logits, np.expand_dims(test_img[num], axis=-1), specific_label)
logger.info(specific_test_accuracy)

logit = np.argmax(test_logit, 1)
indx = np.where(test_predict[:] == 0.)[0]
logger.info("{} / {}".format(len(indx), len(test_img[num])))
nrow = np.ceil(np.sqrt(len(indx))).astype(np.int)
fig, ax = plt.subplots(nrows=nrow, ncols=nrow, figsize = (10, 10))
img = test_img[num][indx]
for i in range(nrow):
    for j in range(nrow):
        if nrow*i+j >= len(indx):
            ax[i][j].axis('off')
            break
        else:
            ax[i][j].imshow(img[nrow*i+j])
            ax[i][j].axis('off')

logit = logit[indx]

logit_class = {}
for num in range(10):
    logit_class[num]= 0
    
for pred in logit:
    logit_class[pred] += 1
for num in range(10):
    if not logit_class[num] == 0:
        logger.info('The number of prediction as {} : {}'.format(num, logit_class[num]))

logger.info("wrong prediction")
for i in range(nrow):
    if nrow*(i+1) < len(indx):
        logger.info(logit[nrow*i : nrow*(i+1)])
    else:
        logger.info(logit[nrow*i : len(indx)])

plt.show()
