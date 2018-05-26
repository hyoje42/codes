# python cutout.py -g 1 -s cifar10_cutout -d cifar10 -b 128 -e 2 -cs 8 --cutout
# python cutout.py -g 1 -s cifar100_baseline -d cifar100 -b 128 -e 2

import tensorflow as tf
import numpy as np
import time
import os
import logging
import argparse

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='cutout')
    parser.add_argument('-g', '--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    
    parser.add_argument('-p', '--path', dest='path', help='path of datasets', 
                        default='/data1/home/hyoje42/datasets')                       
    
    parser.add_argument('-s', '--save', dest='save_model', help='name of model', 
                        default='experiment')
                        
    parser.add_argument('-d', '--dataset', dest='dataset', help='cifar10 or cifar100',
                        default='cifar10')
                        
    parser.add_argument('-b', '--batch', dest='batch_size', help='batch size',
                        default=32, type=int)
                        
    parser.add_argument('-e', '--epochs', dest='epochs', help='total numper of epochs',
                        default=10, type=int)
                            
    parser.add_argument('-cs', '--cutout_size', dest='cutout_size', help='size of patch in cutout',
                        default=8, type=int)
    
    parser.add_argument('-n', '--num_hole', dest='num_hole', help='the number of patchs in cutout',
                        default=1, type=int)
    
    parser.add_argument('-c', '--cutout', dest='is_cutout', help='whether cutout or not',
                        action='store_true')
    args = parser.parse_args()
    
    return args    

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dicts = pickle.load(fo)
    return dicts

def batch_norm(inputs, momentum=0.9, epsilon=1e-5, scale=True, is_training=True, name='batch_norm'):
    return tf.contrib.layers.batch_norm(inputs=inputs, decay=momentum, 
                                        epsilon=epsilon, scale=scale, 
                                        is_training=is_training, 
                                        updates_collections=None, scope=name)

def conv2d(inputs, filters, k_size, reg=False, padding='same', activation=None, strides=1, name='conv'):
    if reg:
        return tf.layers.conv2d(inputs=inputs, filters=filters, kernel_size=k_size, 
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(0.01),
                                strides=strides, padding=padding, 
                                activation=activation, name=name)
    else:
        return tf.layers.conv2d(inputs=inputs, filters=filters, kernel_size=k_size, 
                                strides=strides, padding=padding, 
                                activation=activation, name=name)
    
def maxpool(inputs, k_size=2, padding='same', strides=2, name='pooling'):
    return tf.layers.max_pooling2d(inputs=inputs, pool_size=k_size, 
                                   padding=padding, strides=strides, name=name)
                                   
def block(inputs, filters, training=True, block_num=0, is_bn=True):
    if is_bn:
        h = conv2d(inputs, filters=filters, k_size=3, activation=None, name='h{}'.format(block_num))
        h_bn = batch_norm(h, is_training=training, name='h{}_bn'.format(block_num))
        h_act = tf.nn.relu(h_bn, name='h{}_act'.format(block_num))
        m = maxpool(h_act, name='h{}_pool'.format(block_num))
    else:
        h = conv2d(inputs, filters=filters, k_size=3, activation=tf.nn.relu, name='h{}'.format(block_num))
        m = maxpool(h, name='h{}_pool'.format(block_num))
    return m

def get_accuracy(sess, logits, img, label, batch_size):    
    if len(img) < batch_size:
        logit = sess.run(logits, feed_dict={X: img, training: False})
    else:
        total_batch = int(np.ceil((len(img) / float(batch_size))))
        num_classes = int(logits.shape[-1])
        logit = np.array([], dtype=np.int64).reshape(0, num_classes)
        idxs = range(len(img))
        for i in range(total_batch):
            idxs_i = idxs[i * batch_size : (i + 1) * batch_size]
            feed_dict = {X: img[idxs_i], training: False}
            logit = np.concatenate((logit, sess.run(logits, feed_dict=feed_dict)), axis=0)

    correct_pred = np.equal(np.argmax(logit, 1), label).astype(np.float32)
    accuracy = np.mean(correct_pred)
    
    return accuracy, logit, correct_pred

class Cutout(object):
    def __init__(self, length, n_hole):
        self.length = length
        self.n_hole = n_hole
        
    def act(self, img):
        N, h, w = img.shape[:-1]
        mask = np.ones([N, h, w], np.float32)
        
        for i in range(self.n_hole):
            for idx in range(N):
                
                y = np.random.randint(h)
                x = np.random.randint(w)

                y1 = np.clip(y - self.length / 2, a_min=0, a_max=h).astype('int32')
                y2 = np.clip(y + self.length / 2, a_min=0, a_max=h).astype('int32')
                x1 = np.clip(x - self.length / 2, a_min=0, a_max=w).astype('int32')
                x2 = np.clip(x + self.length / 2, a_min=0, a_max=w).astype('int32')

                mask[idx, y1:y2, x1:x2] = 0
                
        mask = np.tile(np.expand_dims(mask, axis=-1), reps=3)

        img_cutout = img*mask
        
        return img_cutout
    
if __name__ == '__main__':
    
    args = parse_args()
    
    MODEL = args.save_model
    PATH = args.path
    ROW, COL, DEPTH = 32, 32, 3   
    NUM_CLASSES = {'cifar10' : 10, 'cifar100' : 100}
    
    # save log
    
    logger = logging.getLogger('mylogger')
    fileHandler = logging.FileHandler('./{}.log'.format(MODEL), mode='w')
    streamHandler = logging.StreamHandler()

    logger.addHandler(fileHandler)
    logger.addHandler(streamHandler)

    logger.setLevel(logging.DEBUG)
    
    logger.info('Start time : {}'.format(time.asctime()))
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(args.gpu_id)
    
    # load dataset
    
    if args.dataset == 'cifar10':
        trX = np.array([]).reshape([0, ROW*COL*DEPTH])
        trY = np.array([])
        
        for i in range(5):
            dicts = unpickle(os.path.join(PATH, 'cifar10/data_batch_{}'.format(i+1)))
            trX = np.concatenate((trX, dicts[b'data'])).astype(np.float32)
            trY = np.concatenate((trY, dicts[b'labels'])).astype(np.int64)
    
        dicts = unpickle(os.path.join(PATH, './cifar10/test_batch'))
        teY = np.array(dicts[b'labels'], dtype=np.int64)
        
    elif args.dataset == 'cifar100':
        dicts = unpickle(os.path.join(PATH, 'cifar100/train'))
        trX = dicts[b'data'].astype(np.float32)
        
        # coarse_labels (20 classes) or fine_labels (100 classes)
        trY = np.array(dicts[b'fine_labels'], dtype=np.int64)
        dicts = unpickle(os.path.join(PATH, 'cifar100/test'))
        teY = np.array(dicts[b'fine_labels'], dtype=np.int64)
    
    trX = np.reshape(trX, [len(trX), DEPTH, ROW, COL]).transpose([0, 2, 3, 1])
    teX = dicts[b'data'].astype(np.float32)
    teX = np.reshape(teX, [len(teX), DEPTH, ROW, COL]).transpose([0, 2, 3, 1])
    
    trX /= 255
    teX /= 255
    
    # standardization

    for i in range(trX.shape[-1]):
        cal_img = trX[:, :, :, i]
        trX[:, :, :, i] = (cal_img - cal_img.mean()) / (cal_img.std() + 1e-7)
        cal_img = teX[:, :, :, i]
        teX[:, :, :, i] = (cal_img - cal_img.mean()) / (cal_img.std() + 1e-7)
    
    # make graph
    
    X = tf.placeholder(tf.float32, [None, ROW, COL, DEPTH], name='input')
    Y = tf.placeholder(tf.int64, [None], name='label')
    training = tf.placeholder(tf.bool, name='is_training')

    bl0 = block(inputs=X, filters=32, training=training, block_num=0)
    bl1 = block(inputs=bl0, filters=64, training=training, block_num=1)
    bl2 = block(inputs=bl1, filters=128, training=training, block_num=2)
    bl3 = block(inputs=bl2, filters=256, training=training, block_num=3)

    output = bl3

    flat = tf.reshape(output, [-1, int(np.prod(output.shape[1:]))])
    dense1 = tf.layers.dense(flat, units=128, activation=None, name='dense1')
    dense1_bn = batch_norm(dense1, is_training=training, name='dense1_bn')
    act1 = tf.nn.relu(dense1_bn, name='dense1_act')
    logits = tf.layers.dense(act1, units=NUM_CLASSES[args.dataset], activation=None, name='dense2')

    cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

    # hyper prameter
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    CHECK_POINT_DIR = './save_{}'.format(args.save_model)
    IS_SAVE = False
    
    config=tf.ConfigProto(allow_soft_placement=True)
    #config.gpu_options.allocator_type='BFC'
    config.log_device_placement=False
    config.gpu_options.allow_growth=True
    #config.gpu_options.per_process_gpu_memory_fraction = 0.4

    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    # Restore and Save
    if IS_SAVE:
        saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state(CHECK_POINT_DIR)

        if checkpoint and checkpoint.model_checkpoint_path:
            try:
                saver.restore(sess, checkpoint.model_checkpoint_path)
                logger.info("Successfully loaded:", checkpoint.model_checkpoint_path)
            except:
                logger.info("Error on loading old network weights")
        else:
            logger.info("Could not find old network weights")
    else:
        logger.info('Do not save')
        
        
    cutout = Cutout(length=args.cutout_size, n_hole=args.num_hole)

    # train

    logger.info('Start learning time : {}'.format(time.asctime()))
    logger.info('Batch size : {}, Epochs : {}, Save_dir : {}'.format(BATCH_SIZE, EPOCHS, CHECK_POINT_DIR))
    logger.info('gpu : {}, cutout : {}'.format(args.gpu_id, args.is_cutout))
    logger.info('dataset : {}, save : {}'.format(args.dataset, args.save_model) )
    
    logger.info('Learning Started.')

    total_start = time.time()
    max_accuracy = 0.0
    for epoch in range(EPOCHS):
        epoch_start = time.time()
        avg_cost = 0
        total_batch = int(np.ceil(trX.shape[0] / float(BATCH_SIZE)))
        
        # cutout
        if args.is_cutout:
            logger.info('with cutout')
            trX_cutout = cutout.act(trX)
        else:
            logger.info('without cutout')
            trX_cutout = trX
        
        idxs = np.random.permutation(range(len(trX)))
        for i in range(total_batch):
            idxs_i = idxs[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
            feed_dict = {X: trX_cutout[idxs_i], Y: trY[idxs_i], training: True }    
            c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
            avg_cost += c / total_batch
        logger.info('=====================================')
        logger.info('Epoch : {:02d} , Cost : {:.5f}'.format(epoch + 1, avg_cost))
        train_accuracy, _, _ = get_accuracy(sess, logits, trX_cutout, trY, BATCH_SIZE)
        logger.info('Train Accuracy : {:.5f}'.format(train_accuracy))
        test_accuracy, _, _ = get_accuracy(sess, logits, teX, teY, BATCH_SIZE)
        logger.info('Test Accuracy : {:.5f}'.format(test_accuracy))
        logger.info('Elapsed time of an epoch: {:.5f}'.format(time.time() - epoch_start))
        if test_accuracy > max_accuracy:
            max_accuracy = test_accuracy
            logger.info('===========Best accuracy=============')
            logger.info(time.asctime())
            logger.info('Best Accuracy : {:.5f}'.format(max_accuracy))
            logger.info('Elapsed time to get the best accuracy: {:.5f}'.format(time.time() - total_start))
            if IS_SAVE:
                logger.info('===========Saving network with the best accuracy===========')
                if not os.path.exists(CHECK_POINT_DIR):
                    os.makedirs(CHECK_POINT_DIR)
                saver.save(sess, CHECK_POINT_DIR + "/model", global_step=epoch+1)

    logger.info('=====================================')

    logger.info('Final Test Accuracy : {:.5f}'.format(max_accuracy))
    logger.info('Total elapsed time: {:.5f}'.format(time.time() - total_start))    
