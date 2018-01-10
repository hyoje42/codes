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

class layer:
    def __init__(self):
        pass
        
    def block(self, input_, input_depth, curr_depth, depth):
        L1 = conv(input_, input_depth, curr_depth)
        L2 = conv(L1, curr_depth, curr_depth)
        
        if depth==4:
            L3 = conv(L2, curr_depth, curr_depth)
            L4 = conv(L3, curr_depth, curr_depth)
            L4 = tf.nn.max_pool(L4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        else:
            L4 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        
        return L4
        
            
def conv(input_, in_depth, out_depth, filter_size=3, strides=1, padding='SAME'):
    W = tf.Variable(tf.random_normal([filter_size, filter_size, in_depth, out_depth], stddev=0.01))    # (3, 3, channel, filters)
    L = tf.nn.conv2d(input_, W, strides=[1, strides, strides, 1], padding=padding)
    L_BN = tf.layers.batch_normalization(L, axis=3)
    L_BN = tf.nn.relu(L_BN)
    
    return L_BN
    
X = tf.placeholder(tf.float32, [None, row, col, depth])
Y = tf.placeholder(tf.float32, [None, 10])

layers = layer()
tower1 = layers.block(X, depth, init_size, depth=2)
tower2 = layers.block(tower1, init_size, init_size*2, depth=2)
tower3 = layers.block(tower2, init_size*2, init_size*4, depth=4)
#tower4 = layers.block(tower3, init_size*4, init_size*8, depth=4)
#tower5 = layers.block(tower4, init_size*8, init_size*8, depth=4)

flat_shape = 4*4*init_size*4

flat = tf.reshape(tower3, [-1, flat_shape])

W1_fc = tf.Variable(tf.random_normal([flat_shape, init_size*64]))
b1 = tf.Variable(tf.random_normal([init_size*64]))
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