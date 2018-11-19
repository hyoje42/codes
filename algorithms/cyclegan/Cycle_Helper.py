import tensorflow as tf
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from Cycle_Utils import *

REAL_LABEL = 1.0

def batch_norm(inputs, momentum=0.9, epsilon=1e-5, is_training=True, name='batch_norm'):
    return tf.contrib.layers.batch_norm(inputs=inputs, decay=momentum, 
                                        epsilon=epsilon, scale=True, 
                                        is_training=is_training, 
                                        updates_collections=None, scope=name)
                                        
def conv_bn_act(inputs, filters, is_training, is_bn=True, activation=tf.nn.leaky_relu, name='conv_bn_act'):
    h = tf.layers.conv2d(inputs, filters=filters, kernel_size=3, strides=2, padding='same', name=name+'_conv')
    if is_bn:
        h = batch_norm(h, is_training=is_training, name=name+'_bn')
    if activation != None:
        h = activation(h, name=name+'_act')
    return h

def dconv_bn_act(inputs, filters, is_training, is_bn=True, activation=tf.nn.relu, name='dconv_bn_act'):
    h = tf.layers.conv2d_transpose(inputs, filters=filters, kernel_size=3, strides=2, padding='same', name=name+'_dconv')
    if is_bn:
        h = batch_norm(h, is_training=is_training, name=name+'_bn')
    if activation != None:
        h = activation(h, name=name+'_act')
    return h

def DCGAN_discriminator(inputs, init_filters, is_training, name='DCGAN_D'):
    h = conv_bn_act(inputs, filters=init_filters, is_training=is_training, is_bn=False, name=name+'_D_block1')
    h = conv_bn_act(h, filters=init_filters*2, is_training=is_training, name=name+'_D_block2')
    h = conv_bn_act(h, filters=init_filters*4, is_training=is_training, name=name+'_D_block3')
    h = conv_bn_act(h, filters=init_filters*8, is_training=is_training, name=name+'_D_block4')
    output = conv_bn_act(h, filters=1, is_training=is_training, name=name+'_output')
    return output

def DCGAN_generator(inputs, init_filters, is_training, name='DCGAN_G'):
    h = conv_bn_act(inputs, filters=init_filters, is_training=is_training, activation=tf.nn.relu, name=name+'_G_conv_block1')
    h = conv_bn_act(h, filters=init_filters*2, is_training=is_training, activation=tf.nn.relu, name=name+'_G_conv_block2')
    h = conv_bn_act(h, filters=init_filters*4, is_training=is_training, activation=tf.nn.relu, name=name+'_G_conv_block3')
    h = conv_bn_act(h, filters=init_filters*8, is_training=is_training, activation=tf.nn.relu, name=name+'_G_conv_block4')
    h = dconv_bn_act(h, filters=init_filters*4, is_training=is_training, name=name+'_G_dconv_block1')
    h = dconv_bn_act(h, filters=init_filters*2, is_training=is_training, name=name+'_G_dconv_block2')
    h = dconv_bn_act(h, filters=init_filters*1, is_training=is_training, name=name+'_G_dconv_block3')
    h = dconv_bn_act(h, filters=3, is_training=is_training, is_bn=False, activation=tf.nn.tanh, name=name+'_G_dconv_block4')
    
    return h    

def c7s1_k(inputs, k, is_training, activation=tf.nn.relu, is_bn=True, name='c7s1_k'):
    padded = tf.pad(inputs, [[0, 0], [3, 3], [3, 3], [0, 0]], 'REFLECT')
    h = tf.layers.conv2d(padded, filters=k, kernel_size=7, strides=1, padding='valid', name=name+'_conv')
    if is_bn:
        h = batch_norm(h, is_training=is_training, name=name+'_bn')
    if activation != None:
        h = activation(h, name=name+'_act')
    return h    

def dk(inputs, k, is_training, activation=tf.nn.relu, is_bn=True, name='dk'):
    h = tf.layers.conv2d(inputs, filters=k, kernel_size=3, strides=2, padding='same', name=name+'_conv')
    if is_bn:
        h = batch_norm(h, is_training=is_training, name=name+'_bn')
    if activation != None:
        h = activation(h, name=name+'_act')
    return h

def Rk(inputs, k, is_training, name='Rk'):
    padded1 = tf.pad(inputs, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
    conv1 = tf.layers.conv2d(padded1, filters=k, kernel_size=3, strides=1, padding='valid', name=name+'_conv1')
    norm1 = batch_norm(conv1, is_training=is_training, name=name+'_bn1')
    relu1 = tf.nn.relu(norm1, name=name+'_act1')
    
    padded2 = tf.pad(inputs, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
    conv2 = tf.layers.conv2d(padded2, filters=k, kernel_size=3, strides=1, padding='valid', name=name+'_conv2')
    norm2 = batch_norm(conv2, is_training=is_training, name=name+'_bn2')
    
    output = inputs + norm2
    return output

def n_res_blocks(inputs, is_training, n=6):
    depth = inputs.get_shape()[3]
    for i in range(1,n+1):
        output = Rk(inputs, depth, is_training, 'R{}_{}'.format(depth, i))
        inputs = output
    return output

def uk(inputs, k, is_training, is_bn=True, name='uk'):
    u = tf.layers.conv2d_transpose(inputs, filters=k, kernel_size=3, strides=2, padding='same', name=name+'_dconv')
    if is_bn:
        u = batch_norm(u, is_training=is_training, name=name+'_bn')
    u = tf.nn.relu(u, name=name+'_act')
    return u

def Ck(inputs, k, is_training, strids=2, is_bn=True, name='Ck'):
    h = tf.layers.conv2d(inputs, filters=k, kernel_size=4, strides=strids, padding='same', name=name+'_conv')
    if is_bn:
        h = batch_norm(h, is_training=is_training, name=name+'_bn')
    h = tf.nn.leaky_relu(h, name=name+'_lrelu')
    return h

def Cycle_Discriminator(inputs, is_training, init_filters=64, output_act=None):
    C64 = Ck(inputs, k=init_filters*1, is_training=is_training, is_bn=False, name='C64')
    C128 = Ck(C64,   k=init_filters*2, is_training=is_training, name='C128')
    C256 = Ck(C128,  k=init_filters*4, is_training=is_training, name='C256')
    C512 = Ck(C256,  k=init_filters*8, is_training=is_training, name='C512')
    
    output = tf.layers.conv2d(C512, filters=1, kernel_size=4, strides=1, padding='same', name='output')
    if output_act != None:
        output = output_act(output, name='output_act')
    return output

def Cycle_Generator(inputs, is_training, init_filters=64, last_act=tf.nn.tanh, other_act=tf.nn.relu, is_last_bn=False):
    c7s1_64 = c7s1_k(inputs, k=init_filters, is_training=is_training, activation=other_act, name='c7s1_64')
    d128 = dk(c7s1_64, k=init_filters*2, is_training=is_training, activation=other_act, name='d128')
    d256 = dk(d128, k=init_filters*4, is_training=is_training, activation=other_act, name='d256')
    
    if inputs.shape[1] <= 128:
        res_output = n_res_blocks(d256, is_training=is_training, n=6)
    else:
        res_output = n_res_blocks(d256, is_training=is_training, n=9)
    
    u128 = uk(res_output, k=init_filters*2, is_training=is_training, name='u128')
    u64 = uk(u128, k=init_filters*1, is_training=is_training, name='u64')
    
    output = c7s1_k(u64, k=3, is_training=is_training, activation=last_act, is_bn=is_last_bn, name='output')
    
    return output

def discriminator_loss(D, real, fake):
    error_real = tf.reduce_mean(tf.squared_difference(D(real), REAL_LABEL))
    error_fake = tf.reduce_mean(tf.square(D(fake)))
    
    loss = (error_real + error_fake) / 2
    return loss

def generator_loss(D, fake):
    loss = tf.reduce_mean(tf.squared_difference(D(fake), REAL_LABEL))
    
    return loss

def cycle_consistency_loss(G, F, x, y, lamda=10.0):
    forward_loss  = tf.reduce_mean(tf.abs(F(G(x)) - x))
    backward_loss = tf.reduce_mean(tf.abs(G(F(y)) - y))
    loss = lamda * (forward_loss + backward_loss)
    
    return loss