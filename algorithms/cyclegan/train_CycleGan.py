from Cycle_Helper import *

##### config #####
want_size = 128
lr = 0.0002
SAVE_PATH = './save_model_cycle_base/'
img_save_folder = './images/cycle_base/'

# load data
trainX = load_imgs('trainA', want_size=want_size)
trainY = load_imgs('trainB', want_size=want_size)

# convert range [0, 255] to [-1, 1]
trainX = (trainX - 127.5) / 127.5
trainY = (trainY - 127.5) / 127.5

#define graph
class Discriminator():
    def __init__(self, name, is_training, init_filters=64, model='DCGAN'):
        self.name = name
        self.is_training = is_training
        self.reuse = False
        self.init_filters = init_filters
        self.model = model
        
    def __call__(self, inputs):
        with tf.variable_scope(self.name) as scope:
            if self.reuse:
                scope.reuse_variables()
            if self.model == 'DCGAN':
                h = DCGAN_discriminator(inputs, init_filters=self.init_filters, is_training=self.is_training)
            elif self.model == 'Cycle':
                h = Cycle_Discriminator(inputs, init_filters=self.init_filters, is_training=self.is_training, output_act=None)
        
        self.variables = tf.trainable_variables(scope=self.name)
        self.reuse = True
        
        return h

class Generator():
    def __init__(self, name, is_training, init_filters=64, model='DCGAN'):
        self.name = name
        self.is_training = is_training
        self.reuse = False
        self.init_filters = init_filters
        self.model = model
        
    def __call__(self, inputs):
        with tf.variable_scope(self.name) as scope:
            if self.reuse:
                scope.reuse_variables()
            if self.model == 'DCGAN':
                h = DCGAN_generator(inputs, init_filters=self.init_filters, is_training=self.is_training)
            elif self.model == 'Cycle':
                h = Cycle_Generator(inputs, init_filters=self.init_filters, is_training=self.is_training, 
                                    last_act=tf.nn.tanh, other_act=tf.nn.relu, is_last_bn=False)
                
        self.variables = tf.trainable_variables(scope=self.name)
        self.reuse = True
        
        return h
    
    def samples(self, inputs):
        # imgs : [-1, 1]
        imgs = self.__call__(inputs)
        # range of imgs [-1, 1] to [0, 255]
        imgs = tf.map_fn(convert2int_tf, imgs, dtype=tf.uint8)
        
        return imgs

X = tf.placeholder(tf.float32, shape=[None, want_size, want_size, 3], name='X')
Y = tf.placeholder(tf.float32, shape=[None, want_size, want_size, 3], name='Y')
is_training = tf.placeholder(tf.bool, name='is_training')

D_x = Discriminator('D_x', is_training, init_filters=64, model='Cycle')
D_y = Discriminator('D_y', is_training, init_filters=64, model='Cycle')
G = Generator('G', is_training, init_filters=64, model='Cycle')
F = Generator('F', is_training, init_filters=64, model='Cycle')

fake_X = F(Y)
fake_Y = G(X)

cycle_loss = cycle_consistency_loss(G, F, X, Y)
G_loss = generator_loss(D_y, fake_Y) + cycle_loss
F_loss = generator_loss(D_x, fake_X) + cycle_loss
D_x_loss = discriminator_loss(D_x, X, fake_X)
D_y_loss = discriminator_loss(D_y, Y, fake_Y)

G_optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(G_loss, var_list=G.variables)
F_optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(F_loss, var_list=F.variables)
D_x_optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(D_x_loss, var_list=D_x.variables)
D_y_optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(D_y_loss, var_list=D_y.variables)

with tf.control_dependencies([G_optimizer, F_optimizer, D_x_optimizer, D_y_optimizer]):
    all_optimizers = tf.no_op('all_optimizers')
with tf.control_dependencies([G_optimizer, F_optimizer]):
    gen_optimizers = tf.no_op('gen_optimizers')
    
config=tf.ConfigProto(allow_soft_placement=True)
#config.gpu_options.allocator_type='BFC'
config.log_device_placement=False
config.gpu_options.allow_growth=True
#config.gpu_options.per_process_gpu_memory_fraction = 0.4

sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver(max_to_keep=5)

batch_size = 1
train_num = min(len(trainX), len(trainY))
total_iter = 0

for epoch in range(10000):
    idxs = np.random.permutation(range(train_num))
    idxs_gen = np.random.permutation(range(train_num))
    total_batch = np.ceil(train_num / batch_size).astype(np.int32)
    for i in range(total_batch):
        idx_i = idxs[i * batch_size : (i + 1) * batch_size]
        
        _, G_loss_val, F_loss_val, D_x_loss_val, D_y_loss_val = sess.run(
                                                                [all_optimizers, G_loss, F_loss, D_x_loss, D_y_loss],
            feed_dict = {X: trainX[idx_i], Y: trainY[idx_i], is_training : True}
        )
        
        idx_i_gen = idxs_gen[i * batch_size : (i + 1) * batch_size]
        
        _, G_loss_val, F_loss_val, D_x_loss_val, D_y_loss_val = sess.run(
                                                                [gen_optimizers, G_loss, F_loss, D_x_loss, D_y_loss],
            feed_dict = {X: trainX[idx_i_gen], Y: trainY[idx_i_gen], is_training : True}
        )
        
        if total_iter % 200 == 0:
            print(G_loss_val, F_loss_val, D_x_loss_val, D_y_loss_val)
            
        if total_iter % 200 == 0:
            if not os.path.isdir('./images'):
                os.makedirs('./images')
            if not os.path.isdir(img_save_folder):
                os.makedirs(img_save_folder)
            want_visual_num = 25
            visual_idxs_Y = np.random.permutation(range(len(trainY)))
            visual_idxs_X = np.random.permutation(range(len(trainX)))
            
            real_X = trainX[visual_idxs_X][:want_visual_num]
            real_Y = trainY[visual_idxs_Y][:want_visual_num]
            imsave(real_X, total_iter, img_save_folder, name='real_X.png')
            imsave(real_Y, total_iter, img_save_folder, name='real_Y.png')
            
            fake_X_imgs = sess.run(fake_X, feed_dict={Y : real_Y, is_training : False})
            imsave(fake_X_imgs, total_iter, img_save_folder, name='fake_X.png')
            fake_Y_imgs = sess.run(fake_Y, feed_dict={X : real_X, is_training : False})
            imsave(fake_Y_imgs, total_iter, img_save_folder, name='fake_Y.png')
        
        if total_iter % 10000 == 0:
            saver.save(sess, os.path.join(SAVE_PATH, 'model'), global_step=total_iter)
        
        total_iter += 1
        