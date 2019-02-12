from utils import *
from inception_resnet_v2 import *
"""
os.environ['CUDA_VISIBLE_DEVICES'] = str(model.gpu_id)
model.Set_print()
model.Session()
model.Build_network()
model.sess.run(tf.global_variables_initializer())
model.Set_save()
model.Make_dataset()
model.Train()
"""
class ModelBase:
    def __init__(self, 
                 data_path='/data1/group/ncia/kaggle_protein/dataset/',
                 gpu_id=0,
                 want_size=256,
                 batch_size=32,
                 epochs=1000,
                 model_name='model',
                 num_classes=28,
                 threshold=0.15,
                 print_mode='print',
                 quick=0,
                 load_model='load_model'):
    
        self.data_path = data_path
        self.gpu_id = gpu_id
        self.want_size = want_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.model_name = model_name
        self.num_classes = num_classes
        self.threshold = threshold
        self.print_mode = print_mode
        self.quick = quick        
        self.checkpoint_folder = './save_model_{}'.format(self.model_name)
        
        # additional
        self.load_model_name = load_model
        self.load_checkpoint_folder = 'save_model_{}'.format(self.load_model_name)
    
    def Set_print(self):
        if self.print_mode == 'logger':
            logger = logging.getLogger('mylogger')
            fileHandler = logging.FileHandler('train_{}.log'.format(self.model_name), mode='w')
            streamHandler = logging.StreamHandler()
            logger.addHandler(fileHandler)
            logger.addHandler(streamHandler)
            logger.setLevel(logging.DEBUG)
            self._print = logger.debug
            return logger.debug
        else:
            self._print = pprint.pprint
            return pprint.pprint
            
    def Make_dataset(self, mode='train'):
        ## load dataset
        if mode == 'train':
            df = pd.read_csv(os.path.join(self.data_path, 'train.csv'))
            
            df['Id'] = os.path.join(self.data_path, 'train/') + df['Id'].astype(str)
            trX = np.array(df['Id'])
            trY = make_label_one_hot(df, num_classes=28)

            np.random.seed(777)
            val_num = int(len(trX) * 0.1)
            valX, valY, = trX[:val_num], trY[:val_num], 
            trX, trY  = trX[val_num:], trY[val_num:]

            ## quick learning
            if self.quick:
                rand_idx = np.random.permutation(range(len(valX)))
                rand_idx = rand_idx[:self.quick]
                valX, valY = valX[rand_idx], valY[rand_idx]
                
                rand_idx = np.random.permutation(range(len(trX)))
                rand_idx = rand_idx[:self.quick]
                trX, trY = trX[rand_idx], trY[rand_idx]
                
            self._print('{} {} {} {}'.format(trX.shape, trY.shape, valX.shape, valY.shape))
            
            self.trX, self.trY, self.valX, self.valY = trX, trY, valX, valY
        else:
            df = pd.read_csv(os.path.join(self.data_path, 'sample_submission.csv'))
            df['Id'] = os.path.join(self.data_path, 'test/') + df['Id'].astype(str)
            trX = np.array(df['Id'])
            
            self._print(str(trX.shape))
            
            self.trX, self.df = trX, df
            
    def Evaluate_kaggle(self):
        self.Make_dataset(mode='test')
        self.pred_eval = get_f1_score(self.trX, label=[], X=self.X, pred=self.pred,
                                       sess=self.sess, batch_size=self.batch_size, want_size=self.want_size, 
                                       num_classes=self.num_classes, th=self.threshold, is_progress=True)
        
        pred_lists = make_pred_lists(self.pred_eval, forced=False)
        self.df.Id = self.df['Id'].str.replace(os.path.join(self.data_path, 'test/'), '')
        self.df['Predicted'] = pred_lists
        self.df.to_csv('{}.csv'.format(self.model_name), index=False)
       
    def Session(self):
        config = tf.ConfigProto(allow_soft_placement=True)
        config.log_device_placement=False
        config.gpu_options.allow_growth=True
        #config.gpu_options.per_process_gpu_memory_fraction = 0.4
        
        self.sess = tf.Session(config=config)
        return self.sess
        
    def Set_save(self):
        # self.saver = tf.train.Saver(max_to_keep=5, var_list=self.var_list_global)
        self.saver = tf.train.Saver(max_to_keep=5)
    
    def Train(self, mode='Score'):
        """
        Inputs:
            mode : 'Score', calculate f1-score
                   'Accuracy', calcuate accuracy
        """
        # self.Set_print()
        self._print('Calcuate {}'.format(mode))
        var_num = int(np.sum([np.prod(var.shape) for var in tf.trainable_variables()]))
        self._print('The number of variables : {}'.format(var_num))
        # os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu_id)
        # self.Session()
        # self.Build_network()
        # self.sess.run(tf.global_variables_initializer())
        # self.Set_save()
        # self.Make_dataset()
        
        # log results
        self.scores = []
        self.costs = []
        
        max_score = 0
        total_start = time.time()
        for epoch in range(self.epochs):
            epoch_start = time.time()
            total_batch = np.ceil(len(self.trX) / float(self.batch_size)).astype(np.int32)
            idxs = np.random.permutation(range(len(self.trX)))
            avg_cost = 0
            for i in range(total_batch):
                idxs_i = idxs[i * self.batch_size : (i + 1)*self.batch_size]
                imgs, labels = make_imgs(self.trX[idxs_i], self.want_size), self.trY[idxs_i]
                feed_dict = {self.X: imgs, self.Y: labels, self.is_training: True }
                c, _ = self.sess.run([self.cost, self.optimizer], feed_dict=feed_dict)
                avg_cost += c / total_batch
                printProgress(i, total_batch)
            self.costs.append(avg_cost)
            self._print('='*100)
            self._print('Epoch : {:02d}/{:02d} , Cost : {:.5f}'.format(epoch + 1, self.epochs, avg_cost))
            if mode == 'Score':
                val_score, prediction = get_f1_score(self.valX, self.valY, X=self.X, pred=self.pred,
                                                     sess=self.sess, batch_size=self.batch_size, want_size=self.want_size,
                                                     num_classes=self.num_classes, th=self.threshold)
            elif mode == 'Accuracy':
                val_score, prediction = get_accuracy(self.valX, self.valY, X=self.X, logits=self.logits,
                                                     sess=self.sess, batch_size=self.batch_size, want_size=self.want_size)    
            self.scores.append(val_score)
            self._print('Validation {} : {:.5f} / Threshold : {:.2f}'.format(mode, val_score, self.threshold))
            self._print('Elapsed time of an epoch: {:.0f} sec'.format(time.time() - epoch_start))
            if val_score > max_score:
                max_score = val_score
                self.max_score = max_score
                self.prediction = prediction
                self._print('{:=^100}'.format('Best {}'.format(mode)))
                self._print(time.asctime())
                self._print('Best {} : {:.5f}'.format(mode, max_score))
                self._print('Elapsed time to get the best performance: {:.5f}'.format(time.time() - total_start))
                self._print('{:=^100}'.format('Saving network with the best performance'))
                self.saver.save(self.sess, self.checkpoint_folder + "/model", global_step=epoch+1)
            self.sess.run(self.global_step.assign_add(1))
        self._print('Learning is finished...')
        
    def Load_model_for_eval(self, sess):
        for op in sess.graph.get_operations():
            if '/X' in op.name and 'norm' not in op.name:
                self.X = sess.graph.get_tensor_by_name(op.name + ':0')
                self._print(op.name)
            elif 'pred' in op.name:
                self.pred = sess.graph.get_tensor_by_name(op.name + ':0')
                self._print(op.name)
                
class InceptResV2(ModelBase):
    def Build_network(self):
        MEAN, STD = 17.234482, 36.658203
        
        with tf.variable_scope(self.model_name):
            self.X = tf.placeholder(tf.uint8, shape=[None, self.want_size, self.want_size, 4], name='X')
            self.Y = tf.placeholder(tf.float32, shape=[None, self.num_classes], name='Y')
            self.is_training = tf.placeholder_with_default(False, shape=None, name='is_training')

            self.X_norm = tf.divide(tf.cast(self.X, dtype=tf.float32) - tf.constant(value=MEAN), tf.constant(value=STD), name='X_norm')
            
            self.logits, _ = inception_resnet_v2(inputs=self.X_norm, num_classes=self.num_classes, is_training=self.is_training)
            self.pred = tf.nn.sigmoid(self.logits, name='pred')
            
        self.cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.Y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(self.cost)

        self.global_step = tf.Variable(tf.constant(1), trainable=False, name='global_step')
        tf.add_to_collection('global_step', self.global_step)
        
        self.var_list_train = tf.trainable_variables(scope=self.model_name)
        self.var_list_global = tf.global_variables(scope=self.model_name)

class InceptResV2_DF_one(ModelBase):
    def Build_network(self):
        MEAN, STD = 17.234482, 36.658203
        
        with tf.variable_scope(self.model_name):
            self.X = tf.placeholder(tf.uint8, shape=[None, self.want_size, self.want_size, 4], name='X')
            self.Y = tf.placeholder(tf.float32, shape=[None, self.num_classes], name='Y')
            self.is_training = tf.placeholder_with_default(False, shape=None, name='is_training')

            self.X_norm = tf.divide(tf.cast(self.X, dtype=tf.float32) - tf.constant(value=MEAN), tf.constant(value=STD), name='X_norm')
            
            self.logits, _ = inception_resnet_v2(inputs=self.X_norm, num_classes=self.num_classes, is_training=self.is_training)
            self.pred = tf.nn.softmax(self.logits, name='pred')
            
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(self.cost)

        self.global_step = tf.Variable(tf.constant(1), trainable=False, name='global_step')
        tf.add_to_collection('global_step', self.global_step)
        
        self.var_list_train = tf.trainable_variables(scope=self.model_name)
        self.var_list_global = tf.global_variables(scope=self.model_name)
            
    def Make_dataset_DF_one(self, specific):
        self.trY, self.valY = make_df_otherwise(self.trY, specific=specific), make_df_otherwise(self.valY, specific=specific)
        self._print('change label.. defect {} and otherwise'.format(specific))
        self._print('{} {} {} {}'.format(self.trX.shape, self.trY.shape, self.valX.shape, self.valY.shape))

class VggNet_DF_one(ModelBase):
    def Build_network(self, mode='vgg16'):
        MEAN, STD = 17.234482, 36.658203
        
        with tf.variable_scope(self.model_name):
            self.X = tf.placeholder(tf.uint8, shape=[None, self.want_size, self.want_size, 4], name='X')
            self.Y = tf.placeholder(tf.float32, shape=[None, self.num_classes], name='Y')
            self.is_training = tf.placeholder_with_default(False, shape=None, name='is_training')

            self.X_norm = tf.divide(tf.cast(self.X, dtype=tf.float32) - tf.constant(value=MEAN), tf.constant(value=STD), name='X_norm')
            if mode == 'vgg16':
                self.logits = vgg16(inputs=self.X_norm, num_classes=self.num_classes, is_training=self.is_training)
            elif mode == 'vgg19':
                self.logits = vgg19(inputs=self.X_norm, num_classes=self.num_classes, is_training=self.is_training)
                
            self.pred = tf.nn.softmax(self.logits, name='pred')
            
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(self.cost)

        self.global_step = tf.Variable(tf.constant(1), trainable=False, name='global_step')
        tf.add_to_collection('global_step', self.global_step)
        
        self.var_list_train = tf.trainable_variables(scope=self.model_name)
        self.var_list_global = tf.global_variables(scope=self.model_name)
            
    def Make_dataset_DF_one(self, specific):
        self.trY, self.valY = make_df_otherwise(self.trY, specific=specific), make_df_otherwise(self.valY, specific=specific)
        self._print('change label.. defect {} and otherwise'.format(specific))
        self._print('{} {} {} {}'.format(self.trX.shape, self.trY.shape, self.valX.shape, self.valY.shape))        
          
class Ensemble_pretrain_DF0(ModelBase):    
    def Load_meta(self):
        self.load_checkpoint = tf.train.get_checkpoint_state(self.load_checkpoint_folder)
        self.load_model_path = self.load_checkpoint.model_checkpoint_path
        self.load_meta_path = self.load_model_path + '.meta'

        self.load_saver = tf.train.import_meta_graph(self.load_meta_path)
        self.load_var_list_global = tf.global_variables(scope=self.load_model_name)
        self.load_var_list_train = tf.trainable_variables(scope=self.load_model_name)
        
        config = tf.ConfigProto(allow_soft_placement=True)
        config.log_device_placement=False
        config.gpu_options.allow_growth=True
        with tf.Session(config=config) as sess:
            for op in sess.graph.get_operations():
                if self.load_model_name + '/X' == op.name:
                    self.X = sess.graph.get_tensor_by_name(op.name + ':0')
                    self._print(op.name)
                elif self.load_model_name + '/is_training' == op.name:
                    self.is_training = sess.graph.get_tensor_by_name(op.name + ':0')
                    self._print(op.name)
                elif self.load_model_name + '/X_norm' == op.name:
                    self.X_norm = sess.graph.get_tensor_by_name(op.name + ':0')
                    self._print(op.name)
                elif self.load_model_name + '/InceptionResnetV2/Logits/Logits/BiasAdd' == op.name:
                    self.logits_load = sess.graph.get_tensor_by_name(op.name + ':0')
                    self._print(op.name)
    
    def Build_network(self):
        with tf.variable_scope(self.model_name):
            self.Y = tf.placeholder(tf.float32, shape=[None, self.num_classes], name='Y')
            self.logits_DF27, _ = inception_resnet_v2(inputs=self.X_norm, num_classes=self.num_classes-1, is_training=self.is_training)
            self.logits_DF0_split = tf.split(self.logits_load, 2, axis=-1)[0]
            self.logits = tf.concat((self.logits_DF0_split, self.logits_DF27), axis=-1, name='logits_final')
            self.pred = tf.nn.sigmoid(self.logits, name='pred')
            
        self.cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.Y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(self.cost)

        self.global_step = tf.Variable(tf.constant(1), trainable=False, name='global_step')
        tf.add_to_collection('global_step', self.global_step)
        
        self.var_list_train = tf.trainable_variables(scope=self.model_name)
        self.var_list_global = tf.global_variables(scope=self.model_name)

class Ensemble_DF0_scratch(ModelBase):
    def Build_network(self):
        MEAN, STD = 17.234482, 36.658203
        
        with tf.variable_scope(self.model_name):
            self.X = tf.placeholder(tf.uint8, shape=[None, self.want_size, self.want_size, 4], name='X')
            self.Y = tf.placeholder(tf.float32, shape=[None, self.num_classes], name='Y')
            self.is_training = tf.placeholder_with_default(False, shape=None, name='is_training')

            self.X_norm = tf.divide(tf.cast(self.X, dtype=tf.float32) - tf.constant(value=MEAN), tf.constant(value=STD), name='X_norm')
        
        with tf.variable_scope(self.model_name+'_DF0'):    
            self.logits_DF0, _ = inception_resnet_v2(inputs=self.X_norm, num_classes=2, is_training=self.is_training)
            self.pred_DF0 = tf.nn.softmax(self.logits_DF0, name='pred')
        
        self.var_list_train_DF0 = tf.trainable_variables(scope=self.model_name+'_DF0')
        
        cond = self.Y[:, 0] > 0
        self.y_cond = tf.where(cond, x=df_one(self.Y, choose=0), y=df_one(self.Y, choose=1))
        
        self.cost_DF0 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits_DF0, labels=self.y_cond), name='cost_DF0')
        self.optimizer_DF0 = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(self.cost_DF0, var_list=self.var_list_train_DF0)
        
        with tf.variable_scope(self.model_name+'_DF27'):            
            self.logits_DF27, _ = inception_resnet_v2(inputs=self.X_norm, num_classes=27, is_training=self.is_training)
            # self.pred = tf.nn.sigmoid(self.logits, name='pred')
        
        self.var_list_train_DF27 = tf.trainable_variables(scope=self.model_name+'_DF27')
        
        self.logits = tf.concat((tf.split(self.pred_DF0, 2, axis=-1)[0], self.logits_DF27), axis=-1, name=self.model_name + '_final_logits')
        self.pred = tf.nn.sigmoid(self.logits, name='pred')
        
        self.cost_DF27 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.Y), name='cost_DF27')
        self.optimizer_DF27 = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(self.cost_DF27, var_list=self.var_list_train_DF27)
                
        self.global_step = tf.Variable(tf.constant(1), trainable=False, name='global_step')
        tf.add_to_collection('global_step', self.global_step)
        
        with tf.control_dependencies([self.optimizer_DF0, self.optimizer_DF27]):
            self.optimizer = tf.no_op('all_optimizers')
        self.cost = tf.add(self.cost_DF0, self.cost_DF27, name='cost_final')

        self.var_list_global = tf.global_variables(scope=self.model_name)
 
class Ensemble_DF0_scratch_vgg(ModelBase):
    def Build_network(self, mode='vgg16'):
        MEAN, STD = 17.234482, 36.658203
        
        with tf.variable_scope(self.model_name):
            self.X = tf.placeholder(tf.uint8, shape=[None, self.want_size, self.want_size, 4], name='X')
            self.Y = tf.placeholder(tf.float32, shape=[None, self.num_classes], name='Y')
            self.is_training = tf.placeholder_with_default(False, shape=None, name='is_training')

            self.X_norm = tf.divide(tf.cast(self.X, dtype=tf.float32) - tf.constant(value=MEAN), tf.constant(value=STD), name='X_norm')
        
        with tf.variable_scope(self.model_name+'_DF0'):    
            self.logits_DF0, _ = inception_resnet_v2(inputs=self.X_norm, num_classes=2, is_training=self.is_training)
            self.pred_DF0 = tf.nn.softmax(self.logits_DF0, name='pred')
        
        self.var_list_train_DF0 = tf.trainable_variables(scope=self.model_name+'_DF0')
        
        cond = self.Y[:, 0] > 0
        self.y_cond = tf.where(cond, x=df_one(self.Y, choose=0), y=df_one(self.Y, choose=1))
        
        self.cost_DF0 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits_DF0, labels=self.y_cond), name='cost_DF0')
        self.optimizer_DF0 = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(self.cost_DF0, var_list=self.var_list_train_DF0)
        
        with tf.variable_scope(self.model_name+'_DF27'):
            if mode == 'vgg16':
                self.logits_DF27 = vgg16(inputs=self.X_norm, num_classes=27, is_training=self.is_training)
            elif mode == 'vgg19':
                self.logits_DF27 = vgg19(inputs=self.X_norm, num_classes=27, is_training=self.is_training)
            # self.pred = tf.nn.sigmoid(self.logits, name='pred')
        
        self.var_list_train_DF27 = tf.trainable_variables(scope=self.model_name+'_DF27')
        
        self.logits = tf.concat((tf.split(self.pred_DF0, 2, axis=-1)[0], self.logits_DF27), axis=-1, name=self.model_name + '_final_logits')
        self.pred = tf.nn.sigmoid(self.logits, name='pred')
        
        self.cost_DF27 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.Y), name='cost_DF27')
        self.optimizer_DF27 = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(self.cost_DF27, var_list=self.var_list_train_DF27)
                
        self.global_step = tf.Variable(tf.constant(1), trainable=False, name='global_step')
        tf.add_to_collection('global_step', self.global_step)
        
        with tf.control_dependencies([self.optimizer_DF0, self.optimizer_DF27]):
            self.optimizer = tf.no_op('all_optimizers')
        self.cost = tf.add(self.cost_DF0, self.cost_DF27, name='cost_final')

        self.var_list_global = tf.global_variables(scope=self.model_name)
        
class SimpleModel(ModelBase):
    def Build_network(self):
        with tf.variable_scope(self.model_name):
            self.X = tf.placeholder(tf.uint8, shape=[None, self.want_size, self.want_size, 4], name='X')
            self.Y = tf.placeholder(tf.float32, shape=[None, self.num_classes], name='Y')
            self.is_training = tf.placeholder_with_default(False, shape=None, name='is_training')
            
            conv = tf.layers.conv2d(inputs=tf.cast(self.X, dtype=tf.float32), filters=3, kernel_size=3, padding='same')
            dense = tf.layers.dense(inputs=tf.layers.flatten(conv), units=28)
            
            self.logits = dense
            self.pred = tf.nn.sigmoid(self.logits, name='pred')
            
        self.cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.Y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(self.cost)

        self.global_step = tf.Variable(tf.constant(1), trainable=False, name='global_step')
        tf.add_to_collection('global_step', self.global_step)
        
        self.var_list_train = tf.trainable_variables(scope=self.model_name)
        self.var_list_global = tf.global_variables(scope=self.model_name)     

    
