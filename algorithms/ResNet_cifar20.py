from keras.datasets import mnist, cifar10
(img_train, label_train), (img_test, label_test) = cifar10.load_data()
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Input, Add, Flatten, AveragePooling2D
from keras.layers import Activation
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras import losses, optimizers, regularizers
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='ResNet')
parser.add_argument('-e', '--epoch', dest='epochs', help='epochs', default=100, type=int) 
args = parser.parse_args()
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

img_train = img_train - np.mean(img_train, axis=0)
img_test = img_test - np.mean(img_test, axis= 0)
img_train = img_train / np.var(img_train, axis=0)
img_test = img_test / np.var(img_test, axis =0)

label_train = np_utils.to_categorical(label_train, 10)
label_test = np_utils.to_categorical(label_test, 10)

reg_lamda = 1e-2

class layer:
    def __init__(self, input_unit, filter_n, layer_n):
        self.result = input_unit
        self.layer_n = layer_n
        self.filter_n = filter_n
    def add(self, filter_n):
        tower_1 = Conv2D(filter_n, (3, 3), padding='same', kernel_regularizer=regularizers.l2(reg_lamda))(self.result)
        tower_1 = BatchNormalization(axis = 3)(tower_1)
        tower_1 = Activation('relu')(tower_1)
        tower_2 = Conv2D(filter_n, (3, 3), padding='same', kernel_regularizer=regularizers.l2(reg_lamda))(tower_1)
        tower_2 = BatchNormalization(axis = 3)(tower_2)
        add_1 = Add()([tower_2, self.result])
        add_1 = Activation('relu')(add_1)
        
        return add_1
    
    def implement(self, stride=2):
        tower_1 = Conv2D(self.filter_n, (3, 3), strides=stride, padding='same', kernel_regularizer=regularizers.l2(reg_lamda))(self.result)
        tower_1 = BatchNormalization(axis = 3)(tower_1)
        tower_1 = Activation('relu')(tower_1)
        tower_2 = Conv2D(self.filter_n, (3, 3), padding='same', kernel_regularizer=regularizers.l2(reg_lamda))(tower_1)
        tower_2 = BatchNormalization(axis = 3)(tower_2)
        one_by_one = Conv2D(self.filter_n, (1, 1), strides=stride, padding='same', kernel_regularizer=regularizers.l2(reg_lamda))(self.result)
        self.result = Add()([tower_2, one_by_one])
        self.result = Activation('relu')(self.result)
        
        for d in range(self.layer_n - 1):            
            self.result = self.add(self.filter_n)
            
        return self.result


init_n = 16

input1 = Input(shape=(row, col, depth))
input2 = Conv2D(init_n, (3, 3), strides=1, padding='same', kernel_regularizer=regularizers.l2(reg_lamda))(input1)
input2 = BatchNormalization(axis=3)(input2)
input2 = Activation('relu')(input2)
layer1 = layer(input2, init_n, 3)
res1 = layer1.implement(1)
layer2 = layer(res1, init_n*2, 3)
res2 = layer2.implement()
layer3 = layer(res2, init_n*4, 3)
res3 = layer3.implement()
res3 = AveragePooling2D((2, 2), padding='same')(res3)
fc = Flatten()(res3)
fc = Dense(10, activation='softmax')(fc)
model = Model(inputs=input1, outputs=fc)
model.summary()

model.compile(loss=losses.categorical_crossentropy, 
              optimizer=optimizers.sgd(lr=0.01, decay=0.0001, momentum=0.9),
             metrics=['accuracy'])
model.fit(img_train, label_train, batch_size=128, epochs=args.epochs, validation_split=0.1)
score = model.evaluate(img_test, label_test)
print('\nTest score :', score[0])
print('Test accuracy :', score[1])
