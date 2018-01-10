import tensorflow as tf
from keras.datasets import mnist
(img_train, label_train), (img_test, label_test) = mnist.load_data()

import numpy as np
import matplotlib.pyplot as plt
import time
from keras.utils import np_utils

# split test data
test_img = {}
for idx in range(10):
    test_img[idx] = []

for idx in range(len(img_test)):
    test_img[label_test[idx]].append(img_test[idx])
for num in range(10):
    test_img[num] = np.array(test_img[num])

for num in range(10):
    print("shape of {} : {}".format(num, test_img[num].shape))

# split training data
org_image = {}
for idx in range(10):
    org_image[idx] = []

for idx in range(len(img_train)):
    org_image[label_train[idx]].append(img_train[idx])
for num in range(10):
    org_image[num] = np.array(org_image[num])

for num in range(10):
    print("shape of {} : {}".format(num, org_image[num].shape))

image = {}
#helper = [10, 10, 100, 100, 100, 300, 300, 1000, 1000, 1000]
helper = [1000, 1000, 300, 100, 100, 10, 10, 10, 10, 300]

for num in range(10):
    idxs = np.random.permutation(range(len(org_image[num])))
    image[num] = org_image[num][idxs[0:helper[num]]]
    print("The number of {} : {}".format(num, image[num].shape))

# training data - original

img_train = np.concatenate((image[0], image[1]), axis=0)
label_train = np.concatenate((np.full((1000), 0), np.full((1000), 1)), axis=0)

for num in [2, 9]:
    for i in range(3):
        img_train = np.concatenate((img_train, image[num]), axis=0)
        append_label = np.full((len(image[num])), num)
        label_train = np.concatenate((label_train, append_label), axis=0)
for num in [3, 4]:
    for i in range(10):
        img_train = np.concatenate((img_train, image[num]), axis=0)
        append_label = np.full((len(image[num])), num)
        label_train = np.concatenate((label_train, append_label), axis=0)

for num in [5, 6, 7, 8]:
    for i in range(100):
        img_train = np.concatenate((img_train, image[num]), axis=0)
        append_label = np.full((len(image[num])), num)
        label_train = np.concatenate((label_train, append_label), axis=0)

idxs = np.random.permutation(range(len(img_train)))
img_train = img_train[idxs]
label_train = label_train[idxs]
        
print(img_train.shape)
print(label_train.shape)

# training data - major

img_train = np.concatenate((image[0], image[1]), axis=0)
label_train = np.concatenate((np.full((1000), 0), np.full((1000), 1)), axis=0)

for num in [2, 9]:
    for i in range(3):
        img_train = np.concatenate((img_train, image[num]), axis=0)
        append_label = np.full((len(image[num])), num)
        label_train = np.concatenate((label_train, append_label), axis=0)
for num in [3, 4]:
    for i in range(10):
        img_train = np.concatenate((img_train, image[num]), axis=0)
        append_label = np.full((len(image[num])), num)
        label_train = np.concatenate((label_train, append_label), axis=0)

idxs = np.random.permutation(range(len(img_train)))
img_train = img_train[idxs]
label_train = label_train[idxs]
        
print(img_train.shape)
print(label_train.shape)

# training data - minor

img_train = np.concatenate((image[5], image[6]), axis=0)
label_train = np.concatenate((np.full((len(image[5])), 5), np.full((len(image[6])), 6)), axis=0)
for num in [5, 6, 7, 8]:
    for i in range(100):
        img_train = np.concatenate((img_train, image[num]), axis=0)
        append_label = np.full((len(image[num])), num)
        label_train = np.concatenate((label_train, append_label), axis=0)

idxs = np.random.permutation(range(len(img_train)))
img_train = img_train[idxs]
label_train = label_train[idxs]
        
print(img_train.shape)
print(label_train.shape)

# training data - minor2

img_train = np.concatenate((image[2], image[9]), axis=0)
label_train = np.concatenate((np.full((len(image[2])), 2), np.full((len(image[9])), 9)), axis=0)

for num in [2, 9]:
    for i in range(2):
        img_train = np.concatenate((img_train, image[num]), axis=0)
        append_label = np.full((len(image[num])), num)
        label_train = np.concatenate((label_train, append_label), axis=0)
for num in [3, 4]:
    for i in range(10):
        img_train = np.concatenate((img_train, image[num]), axis=0)
        append_label = np.full((len(image[num])), num)
        label_train = np.concatenate((label_train, append_label), axis=0)

for num in [5, 6, 7, 8]:
    for i in range(100):
        img_train = np.concatenate((img_train, image[num]), axis=0)
        append_label = np.full((len(image[num])), num)
        label_train = np.concatenate((label_train, append_label), axis=0)

idxs = np.random.permutation(range(len(img_train)))
img_train = img_train[idxs]
label_train = label_train[idxs]
        
print(img_train.shape)
print(label_train.shape)


# plot and verify the training data
fig, ax = plt.subplots(nrows=10, ncols=10, figsize = (10, 10))
check = 2200
for i in range(10):
    for j in range(10):
        ax[i][j].imshow(img_train[10*i+j + check])

    print(label_train[10*i + check : 10*(i+1) + check])
        
plt.show()


# setting the training data

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

# Original

X = tf.placeholder(tf.float32, [None, img_row, img_col, img_depth])
Y = tf.placeholder(tf.float32, [None, 10])
training = tf.placeholder(tf.bool)

with tf.variable_scope('conv_layer'):
    conv1 = tf.layers.conv2d(inputs=X, filters=32, kernel_size=[3, 3],
                            padding='SAME', activation=tf.nn.relu, name='conv1')
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], 
                                    padding='SAME', strides=2, name='pool1')
    conv2 = tf.layers.conv2d(inputs=pool1, filters=32, kernel_size=[3, 3],
                            padding='SAME', activation=tf.nn.relu, name='conv2')
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], 
                                    padding='SAME', strides=2, name='pool2')
    conv3 = tf.layers.conv2d(inputs=pool2, filters=32, kernel_size=[3, 3],
                            padding='SAME', activation=tf.nn.relu, name='conv3')
    pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], 
                                padding='SAME', strides=2, name='pool3')
with tf.variable_scope('fc_layer'):
    flat = tf.reshape(pool3, [-1, 4*4*32], name='flat')
    dense1 = tf.layers.dense(inputs=flat, units=1024, activation=tf.nn.relu, 
                             name='dense1')
    drop1 = tf.layers.dropout(inputs=dense1, rate=0.5, training=training)
    dense2 = tf.layers.dense(inputs=drop1, units=512, activation=tf.nn.relu, 
                             name='dense2')
    drop2 = tf.layers.dropout(inputs=dense2, rate=0.5, training=training)
    logits = tf.layers.dense(inputs=drop2, units=10, name='logit')

# load the pretrained weights
W = np.load('./Weights/weights_original_epcoh1000.npy')

# initialized by original Conv + FC

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
                            kernel_initializer=tf.constant_initializer(W[10]), bias_initializer=tf.constant_initializer(W[11]), name='logit')


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
print ('Test Accuracy : {:.5f}'.format(test_accuracy))



# training - original

batch_size = 256
epochs = 1000

print ('Learning Started.')

sess = tf.Session()
sess.run(tf.global_variables_initializer())

total_start = time.time()
for epoch in range(epochs):
    start = time.time()
    avg_cost = 0
    total_batch = int(img_train.shape[0] / batch_size)

    for i in range(total_batch):
        idxs = np.random.permutation(range(len(img_train)))
        idxs_i = idxs[i * batch_size: (i + 1) * batch_size]
        feed_dict = {X: img_train[idxs_i], Y: label_train[idxs_i], training: True}
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost += c / total_batch
    if (epoch+1) % 100 == 0:
        print ('Epoch : {:02d} , Cost : {:.5f}'.format(epoch + 1, avg_cost))
        train_accuracy, _, _ = get_accuracy(sess, logits, img_train, label_train)
        print ('Train Accuracy : {:.5f}'.format(train_accuracy))
        test_accuracy, _, _ = get_accuracy(sess, logits, img_test, label_test)
        print ('Test Accuracy : {:.5f}'.format(test_accuracy))
        print ('Elapsed time : {:.5f}'.format(time.time() - start))
        weights = tf.trainable_variables()
        weights = sess.run(weights)
        np.save('./Weights/weights_original_epcoh{}.npy'.format(epoch+1), weights)

print ('Total elapsed time : {:.5f}'.format(time.time() - total_start))

# training - major

batch_size = 256
epochs = 1000

print ('Learning Started.')

sess = tf.Session()
sess.run(tf.global_variables_initializer())

total_start = time.time()
for epoch in range(epochs):
    start = time.time()
    avg_cost = 0
    total_batch = int(img_train.shape[0] / batch_size)

    for i in range(total_batch):
        idxs = np.random.permutation(range(len(img_train)))
        idxs_i = idxs[i * batch_size: (i + 1) * batch_size]
        feed_dict = {X: img_train[idxs_i], Y: label_train[idxs_i], training: True}
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost += c / total_batch
    if (epoch+1) % 100 == 0:
        print ('Epoch : {:02d} , Cost : {:.5f}'.format(epoch + 1, avg_cost))
        train_accuracy, _, _ = get_accuracy(sess, logits, img_train, label_train)
        print ('Train Accuracy : {:.5f}'.format(train_accuracy))
        test_accuracy, _, _ = get_accuracy(sess, logits, img_test, label_test)
        print ('Test Accuracy : {:.5f}'.format(test_accuracy))
        print ('Elapsed time : {:.5f}'.format(time.time() - start))
        weights = tf.trainable_variables()
        weights = sess.run(weights)
        np.save('./Weights/weights_major_epcoh{}.npy'.format(epoch+1), weights)

print ('Total elapsed time : {:.5f}'.format(time.time() - total_start))



# transfer training - Conv1 + FC1

batch_size = 256
epochs = 1000

print ('Learning Started.')

sess = tf.Session()
sess.run(tf.global_variables_initializer())

total_start = time.time()
for epoch in range(epochs):
    start = time.time()
    avg_cost = 0
    total_batch = int(img_train.shape[0] / batch_size)

    for i in range(total_batch):
        idxs = np.random.permutation(range(len(img_train)))
        idxs_i = idxs[i * batch_size: (i + 1) * batch_size]
        feed_dict = {X: img_train[idxs_i], Y: label_train[idxs_i], training: True}
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost += c / total_batch

    if (epoch+1) % 100 == 0:
        print ('Epoch : {:02d} , Cost : {:.5f}'.format(epoch + 1, avg_cost))
        train_accuracy, _, _ = get_accuracy(sess, logits, img_train, label_train)
        print ('Train Accuracy : {:.5f}'.format(train_accuracy))
        test_accuracy, _, _ = get_accuracy(sess, logits, img_test, label_test)
        print ('Test Accuracy : {:.5f}'.format(test_accuracy))
        print ('Elapsed time : {:.5f}'.format(time.time() - start))
        weights = tf.trainable_variables()
        weights = sess.run(weights)
        np.save('./Weights/weights_transfer_Conv1_Fc1_minor2_epcoh{}.npy'.format(epoch+1), weights)

print ('Total elapsed time : {:.5f}'.format(time.time() - total_start))

# accuracy of test data

for num in range(10):
    specific_label = np_utils.to_categorical(np.full((len(test_img[num])), num), 10)
    specific_test_accuracy, test_predict, test_logit = get_accuracy(sess, logits, np.expand_dims(test_img[num], axis=-1), specific_label)
    print("The specific accurcy of {} : {:.5f}".format(num, specific_test_accuracy))
acc, _, _ = get_accuracy(sess, logits, img_test, label_test)
print("Total accuracy : {:.5f}".format(acc))

num = 5
specific_label = np_utils.to_categorical(np.full((len(test_img[num])), num), 10)
specific_test_accuracy, test_predict, test_logit = get_accuracy(sess, logits, np.expand_dims(test_img[num], axis=-1), specific_label)
print(specific_test_accuracy)

logit = np.argmax(test_logit, 1)
indx = np.where(test_predict[:] == 0.)[0]
print("{} / {}".format(len(indx), len(test_img[num])))
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
        print('The number of prediction as {} : {}'.format(num, logit_class[num]))

print("wrong prediction")
for i in range(nrow):
    if nrow*(i+1) < len(indx):
        print(logit[nrow*i : nrow*(i+1)])
    else:
        print(logit[nrow*i : len(indx)])

plt.show()

num = 6
specific_label = np_utils.to_categorical(np.full((len(test_img[num])), num), 10)
specific_test_accuracy, test_predict, test_logit = get_accuracy(sess, logits, np.expand_dims(test_img[num], axis=-1), specific_label)
print(specific_test_accuracy)

logit = np.argmax(test_logit, 1)
indx = np.where(test_predict[:] == 0.)[0]
print("{} / {}".format(len(indx), len(test_img[num])))
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
        print('The number of prediction as {} : {}'.format(num, logit_class[num]))

print("wrong prediction")
for i in range(nrow):
    if nrow*(i+1) < len(indx):
        print(logit[nrow*i : nrow*(i+1)])
    else:
        print(logit[nrow*i : len(indx)])

plt.show()



num = 7
specific_label = np_utils.to_categorical(np.full((len(test_img[num])), num), 10)
specific_test_accuracy, test_predict, test_logit = get_accuracy(sess, logits, np.expand_dims(test_img[num], axis=-1), specific_label)
print(specific_test_accuracy)

logit = np.argmax(test_logit, 1)
indx = np.where(test_predict[:] == 0.)[0]
print("{} / {}".format(len(indx), len(test_img[num])))
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
        print('The number of prediction as {} : {}'.format(num, logit_class[num]))

print("wrong prediction")
for i in range(nrow):
    if nrow*(i+1) < len(indx):
        print(logit[nrow*i : nrow*(i+1)])
    else:
        print(logit[nrow*i : len(indx)])

plt.show()

num = 8
specific_label = np_utils.to_categorical(np.full((len(test_img[num])), num), 10)
specific_test_accuracy, test_predict, test_logit = get_accuracy(sess, logits, np.expand_dims(test_img[num], axis=-1), specific_label)
print(specific_test_accuracy)

logit = np.argmax(test_logit, 1)
indx = np.where(test_predict[:] == 0.)[0]
print("{} / {}".format(len(indx), len(test_img[num])))
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
        print('The number of prediction as {} : {}'.format(num, logit_class[num]))

print("wrong prediction")
for i in range(nrow):
    if nrow*(i+1) < len(indx):
        print(logit[nrow*i : nrow*(i+1)])
    else:
        print(logit[nrow*i : len(indx)])

plt.show()



num = 9
specific_label = np_utils.to_categorical(np.full((len(test_img[num])), num), 10)
specific_test_accuracy, test_predict, test_logit = get_accuracy(sess, logits, np.expand_dims(test_img[num], axis=-1), specific_label)
print(specific_test_accuracy)

logit = np.argmax(test_logit, 1)
indx = np.where(test_predict[:] == 0.)[0]
print("{} / {}".format(len(indx), len(test_img[num])))
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
        print('The number of prediction as {} : {}'.format(num, logit_class[num]))

print("wrong prediction")
for i in range(nrow):
    if nrow*(i+1) < len(indx):
        print(logit[nrow*i : nrow*(i+1)])
    else:
        print(logit[nrow*i : len(indx)])

plt.show()

