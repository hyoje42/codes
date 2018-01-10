from keras.datasets import mnist, cifar10
(img_train, label_train), (img_test, label_test) = cifar10.load_data()
from keras.utils import np_utils

row, col = img_train.shape[1], img_train.shape[2]

if len(img_train.shape) == 4 :
    depth = img_train.shape[3]
else:
    depth = 1
    img_train = img_train.reshape(img_train.shape[0], row, col, depth)
    img_test = img_test.reshape(img_test.shape[0], row, col, depth)

img_train = img_train.astype('float32')
img_test = img_test.astype('float32')
img_train /= 255
img_test /= 255

label_train = np_utils.to_categorical(label_train, 10)
label_test = np_utils.to_categorical(label_test, 10)

import tensorflow as tf
import numpy as np
import time

init_size = 2
batch_size = 256
learning_rate = 0.001
epochs = 1000

X = tf.placeholder(tf.float32, [None, row, col, depth])
Y = tf.placeholder(tf.float32, [None, label_test.shape[1]])

with tf.variable_scope('conv1_1') as scope:
    W1 = tf.get_variable("W", shape=[3, 3, depth, init_size], 
                         initializer=tf.contrib.layers.xavier_initializer())
    L1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME')
    L_BN1 = tf.layers.batch_normalization(L1, axis=3)
    L_BN1 = tf.nn.relu(L_BN1)

with tf.variable_scope('conv1_2') as scope:
    W2 = tf.get_variable("W", shape=[3, 3, init_size, init_size], 
                         initializer=tf.contrib.layers.xavier_initializer())
    L2 = tf.nn.conv2d(L_BN1, W2, strides=[1, 1, 1, 1], padding='SAME')
    L_BN2 = tf.layers.batch_normalization(L2, axis=3)
    L_BN2 = tf.nn.relu(L_BN2)
    L_BN2 = tf.nn.max_pool(L_BN2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

with tf.variable_scope('conv2_1') as scope:
    W3 = tf.get_variable("W", shape=[3, 3, init_size, init_size*2], 
                         initializer=tf.contrib.layers.xavier_initializer())
    L3 = tf.nn.conv2d(L_BN2, W3, strides=[1, 1, 1, 1], padding='SAME')
    L_BN3 = tf.layers.batch_normalization(L3, axis=3)
    L_BN3 = tf.nn.relu(L_BN3)

with tf.variable_scope('conv2_2') as scope:
    W4 = tf.get_variable("W", shape=[3, 3, init_size*2, init_size*2], 
                         initializer=tf.contrib.layers.xavier_initializer())
    L4 = tf.nn.conv2d(L_BN3, W4, strides=[1, 1, 1, 1], padding='SAME')
    L_BN4 = tf.layers.batch_normalization(L4, axis=3)
    L_BN4 = tf.nn.relu(L_BN4)
    L_BN4 = tf.nn.max_pool(L_BN4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    
with tf.variable_scope('conv3_1') as scope:
    W5 = tf.get_variable("W", shape=[3, 3, init_size*2, init_size*4], 
                         initializer=tf.contrib.layers.xavier_initializer())
    L5 = tf.nn.conv2d(L_BN4, W5, strides=[1, 1, 1, 1], padding='SAME')
    L_BN5 = tf.layers.batch_normalization(L5, axis=3)
    L_BN5 = tf.nn.relu(L_BN5)

with tf.variable_scope('conv3_2') as scope:
    W6 = tf.get_variable("W", shape=[3, 3, init_size*4, init_size*4], 
                         initializer=tf.contrib.layers.xavier_initializer())
    L6 = tf.nn.conv2d(L_BN5, W6, strides=[1, 1, 1, 1], padding='SAME')
    L_BN6 = tf.layers.batch_normalization(L6, axis=3)
    L_BN6 = tf.nn.relu(L_BN6)
    
with tf.variable_scope('conv3_3') as scope:
    W7 = tf.get_variable("W", shape=[3, 3, init_size*4, init_size*4], 
                         initializer=tf.contrib.layers.xavier_initializer())
    L7 = tf.nn.conv2d(L_BN6, W7, strides=[1, 1, 1, 1], padding='SAME')
    L_BN7 = tf.layers.batch_normalization(L7, axis=3)
    L_BN7 = tf.nn.relu(L_BN7)

with tf.variable_scope('conv3_4') as scope:
    W8 = tf.get_variable("W", shape=[3, 3, init_size*4, init_size*4], 
                         initializer=tf.contrib.layers.xavier_initializer())
    L8 = tf.nn.conv2d(L_BN7, W8, strides=[1, 1, 1, 1], padding='SAME')
    L_BN8 = tf.layers.batch_normalization(L8, axis=3)
    L_BN8 = tf.nn.relu(L_BN8)
    L_BN8 = tf.nn.max_pool(L_BN8, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    

flat_shape = 4*4*init_size*4

flat = tf.reshape(L_BN8, [-1, flat_shape])

with tf.variable_scope('fc1') as scope:
    W1_fc = tf.get_variable("W", shape=[flat_shape, init_size*64], 
                         initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.Variable(tf.random_normal([init_size*64]))
    L9 = tf.matmul(flat, W1_fc) + b1
    L9 = tf.layers.batch_normalization(L9, axis=-1)
    L9 = tf.nn.relu(L9)
    L9 = tf.nn.dropout(L9, 

logits = tf.matmul(flat, W1_fc) + b1
logits = tf.layers.batch_normalization(logits, axis=-1)
logits = tf.nn.relu(logits)

W2_fc = tf.Variable(tf.random_normal([init_size*64, init_size*64]))
b2 = tf.Variable(tf.random_normal([init_size*64]))
logits = tf.matmul(logits, W2_fc) + b2
logits = tf.layers.batch_normalization(logits, axis=-1)
logits = tf.nn.relu(logits)

W3_fc = tf.Variable(tf.random_normal([init_size*64, label_train.shape[1]]))
b3 = tf.Variable(tf.random_normal([label_train.shape[1]]))
logits = tf.matmul(logits, W3_fc) + b3

cost = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

print 'Learning Strated.'
for epoch in range(epochs):
    start = time.time()
    avg_cost = 0
    total_batch = int(img_train.shape[0] / batch_size)
    
    for i in range(total_batch):
        idxs = np.random.permutation(xrange(len(img_train)))
        idxs_i = idxs[i * batch_size : (i + 1) * batch_size]
        feed_dict = {X: img_train[idxs_i], Y: label_train[idxs_i]}
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost += c / total_batch
        
    end = time.time()
    print '================================'
    print 'Epoch:{0:04d}'.format(epoch + 1), 'cost ={0:0.4f}'.format(avg_cost)
    print 'Train Accuracy:{0:0.4f}'.format(sess.run(accuracy, feed_dict={X: img_train, Y: label_train}))
    print 'Test Accuracy:{0:0.4f}'.format(sess.run(accuracy, feed_dict={X: img_test, Y: label_test}))
    print 'Elapsed time: {0:0.4f} sec'.format(end - start)

print 'Final Accuracy:{0:0.4f}'.format(sess.run(accuracy, feed_dict={X: img_test, Y: label_test}))