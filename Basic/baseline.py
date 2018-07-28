# python baseline.py -g 1 -s cifar100_baseline -d cifar100 -b 128 -e 100
# python baseline.py -g 1 -s cifar10_standarz -d cifar10 -b 128 -e 100 --standarz 1


import tensorflow as tf
import numpy as np
import time
import os
import logging
import argparse

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='baseline')
    parser.add_argument('-g', '--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    
    parser.add_argument('-p', '--path', dest='path', help='path of datasets', 
                        default='/data1/home/hyoje42/datasets')                       
    
    parser.add_argument('-d', '--dataset', dest='dataset', help='cifar10 or cifar100',
                        default='cifar10')
                        
    parser.add_argument('-b', '--batch_size', dest='batch_size', help='batch size',
                        default=32, type=int)
                        
    parser.add_argument('-e', '--epochs', dest='epochs', help='total numper of epochs',
                        default=10, type=int)
                        
    parser.add_argument('-std', '--standarz', dest='is_standarz', help='whether standardization or not',
                        default=0, type=int)
                        #action='store_true')
    parser.add_argument('-bn', '--batch_norm', dest='batch_norm', help='use batch norm or not',
                        default=0, type=int)                           
                        
    parser.add_argument('-is', '--is_save', dest='is_save', help='save or not',
                        default=1, type=int)                           
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
    
if __name__ == '__main__':
    
    args = parse_args()
    
    # make save name
    HELPER = [args.is_standarz, args.batch_norm]
    HELPER_NAME = ['std', 'bn']
    NAME = args.dataset
    for i in range(2):
        if HELPER[i]:
            NAME += '_{}'.format(HELPER_NAME[i])
    LOG_NAME = './{}.log'.format(NAME)
    FINAL_NAME = NAME
    idx = 2
    while(os.path.isfile(LOG_NAME)):
        FINAL_NAME = NAME
        FINAL_NAME += '_' + str(idx)
        LOG_NAME = './{}.log'.format(FINAL_NAME)
        idx += 1    

    MODEL = FINAL_NAME
    PATH = args.path
    ROW, COL, DEPTH = 32, 32, 3   
    NUM_CLASSES = {'cifar10' : 10, 'cifar100' : 100}
    
    # save log
    
    logger = logging.getLogger('mylogger')
    fileHandler = logging.FileHandler(LOG_NAME, mode='w')
    streamHandler = logging.StreamHandler()
    logger.addHandler(fileHandler)
    logger.addHandler(streamHandler)
    logger.setLevel(logging.DEBUG)
    
    logger.info('Start time : {}'.format(time.asctime()))
    logger.info(str(args))

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
    
    # split validation data
    np.random.seed(777)

    idxs = np.random.permutation(range(len(trX)))
    num_val = int(np.floor(len(trX) * 0.1))
    tr_idxs = idxs[num_val:]
    val_idxs = idxs[:num_val]

    valX = trX[val_idxs]
    valY = trY[val_idxs]
    trX = trX[tr_idxs]
    trY = trY[tr_idxs]

    np.random.seed(int(time.time()))
    
    # standardization
    
    if args.is_standarz:
        logger.info('feature standardization...')
        for i in range(trX.shape[-1]):
            cal_img = trX[:, :, :, i]
            trX[:, :, :, i] = (cal_img - cal_img.mean()) / (cal_img.std() + 1e-7)
            cal_img = teX[:, :, :, i]
            teX[:, :, :, i] = (cal_img - cal_img.mean()) / (cal_img.std() + 1e-7)
    else:
        logger.info('DO NOT feature standardization...')
    
    # make graph
    
    X = tf.placeholder(tf.float32, [None, ROW, COL, DEPTH], name='input')
    Y = tf.placeholder(tf.int64, [None], name='label')
    training = tf.placeholder(tf.bool, name='is_training')

    conv = block(inputs=X, filters=32, training=training, block_num=0, is_bn=args.batch_norm)
    conv = block(inputs=conv, filters=64, training=training, block_num=1, is_bn=args.batch_norm)
    conv = block(inputs=conv, filters=128, training=training, block_num=2, is_bn=args.batch_norm)
    conv = block(inputs=conv, filters=256, training=training, block_num=3, is_bn=args.batch_norm)

    output = conv
    print(output)

    flat = tf.layers.flatten(output, name='flat')
    dense = tf.layers.dense(flat, units=128, activation=None, name='dense1')
    if args.batch_norm:
        dense = batch_norm(dense, is_training=training, name='dense1_bn')
    print(dense)
    act = tf.nn.relu(dense, name='dense1_act')
    logits = tf.layers.dense(act, units=NUM_CLASSES[args.dataset], activation=None, name='dense2')

    cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

    # hyper prameter
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    CHECK_POINT_DIR = './save_{}'.format(MODEL)
    IS_SAVE = args.is_save
    
    config=tf.ConfigProto(allow_soft_placement=True)
    #config.gpu_options.allocator_type='BFC'
    config.log_device_placement=False
    config.gpu_options.allow_growth=True
    #config.gpu_options.per_process_gpu_memory_fraction = 0.4

    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    # Restore and Save
    if IS_SAVE:
        saver = tf.train.Saver(max_to_keep=1)
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
        

    logger.info('Start learning time : {}'.format(time.asctime()))
    logger.info('Batch size : {}, Epochs : {}, Save_dir : {}'.format(BATCH_SIZE, EPOCHS, CHECK_POINT_DIR))
    logger.info('gpu : {}, standardization : {}, batch norm : {}'.format(args.gpu_id, 
                                                                         bool(args.is_standarz),
                                                                         bool(args.batch_norm)))
    logger.info('dataset : {}, save : {}'.format(args.dataset, MODEL) )
    
    # train

    logger.info(time.asctime())
    logger.info('Learning Started.')

    total_start = time.time()
    max_accuracy = 0.0
    val_acc_set = []
    te_acc_set = []
    for epoch in range(EPOCHS):
        epoch_start = time.time()
        avg_cost = 0
        total_batch = np.ceil(trX.shape[0] / BATCH_SIZE).astype(np.int32)
        
        idxs = np.random.permutation(range(len(trX)))
        for i in range(total_batch):
            idxs_i = idxs[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
            feed_dict = {X: trX[idxs_i], Y: trY[idxs_i], training: True }    
            c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
            avg_cost += c / total_batch
        val_accuracy, _, _ = get_accuracy(sess, logits, valX, valY, BATCH_SIZE)
        
        if (epoch % 10) == 0:
            train_accuracy, _, _ = get_accuracy(sess, logits, trX, trY, BATCH_SIZE)
            logger.info('Epoch : {:02d} , Cost : {:.5f}, Val Acc : {:.5f}, Elapsed time : {:5.1f}, Train Acc : {:.5f}'.format(epoch + 1, 
                                                            avg_cost, val_accuracy, time.time() - epoch_start, train_accuracy))
        else:
            logger.info('Epoch : {:02d} , Cost : {:.5f}, Val Acc : {:.5f}, Elapsed time : {:5.1f}'.format(epoch + 1, 
                                                            avg_cost, val_accuracy, time.time() - epoch_start))
        
        if val_accuracy > max_accuracy:
            max_accuracy = val_accuracy
            
            logger.info('{:=^68}'.format('Best accuracy'))
            logger.info(time.asctime())
            logger.info('Elapsed time to get the best accuracy: {:.1f}'.format(time.time() - total_start))
            
            if IS_SAVE:
                #logger.info('{:=^68}'.format('Saving network with the best accuracy'))
                logger.info('Best Accuracy : {:.5f} / Saving models...'.format(max_accuracy))
                if not os.path.exists(CHECK_POINT_DIR):
                    os.makedirs(CHECK_POINT_DIR)
                saver.save(sess, CHECK_POINT_DIR + "/model", global_step=epoch+1)
            else:
                logger.info('Best Accuracy : {:.5f}'.format(max_accuracy))
            te_accuracy, _, _ = get_accuracy(sess, logits, teX, teY, BATCH_SIZE)
            
            # log val / te accuracy
            te_acc_set.append(te_accuracy)
            val_acc_set.append(val_accuracy)
            logger.info('='*68)

    logger.info('='*68)

    train_accuracy, _, _ = get_accuracy(sess, logits, trX, trY, BATCH_SIZE)

    logger.info('Final Train Accuracy : {:.5f}'.format(train_accuracy))
    logger.info('Final Validation Accuracy : {:.5f}'.format(max_accuracy))
    logger.info('Test Accuracy with Best model : {:.5f}'.format(te_acc_set[-1]))
    logger.info('Total Elapsed time : {:.1f}'.format(time.time() - total_start))
    logger.info('End time : {}'.format(time.asctime()))
    
    logger.info(str(val_acc_set))
    logger.info(str(te_acc_set)) 
