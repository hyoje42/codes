from keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
#print(x_train[0])
#print(y_train[0])
#type(x_train)

import numpy as np
import keras
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Input, Activation
from keras.layers import Conv2D, AveragePooling2D, BatchNormalization, GlobalAveragePooling2D
from keras import optimizers
from keras import regularizers

y_train = keras.utils.to_categorical(y_train, num_classes=10)
y_test = keras.utils.to_categorical(y_test, num_classes=10)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

x_train = x_train - np.mean(x_train, axis=0)
x_test = x_test - np.mean(x_test, axis= 0)

x_train = x_train / np.var(x_train, axis=0)
x_test = x_test / np.var(x_test, axis =0)

print(y_test[0])
print(y_test.shape)

def Conv2d_bn(input_tensor, filters):
    
    x = BatchNormalization(axis=3)(input_tensor)
    x = Activation('relu')(x)
    x = Conv2D(filters, kernel_size = (3,3), strides= (1,1), padding='same', kernel_regularizer = regularizers.l2(1e-4))(x)
    x = Dropout(0.2)(x)
    
    return x

def Dense_block(input_tensor, num_layers, growth_rate):
    
    concat_layer = input_tensor
    
    for i in range(num_layers):
        x = Conv2d_bn(concat_layer, growth_rate)
        concat_layer = keras.layers.concatenate([concat_layer, x], axis= -1)
        
    return concat_layer
    


# input: 32x32 images with 3 channels -> (32, 32, 3) tensors.

num_layers = 12
growth_rate = 12

inputs = Input(shape=(32,32,3))

x = Conv2D(16, kernel_size= (3,3), strides= (1,1), padding='same', kernel_regularizer = regularizers.l2(1e-4))(inputs)

##  Dense block 1
x = Dense_block(x, num_layers=num_layers, growth_rate=growth_rate)

x = BatchNormalization(axis=3)(x)
num_feature = int(x.shape[3])
x = Conv2D(filters= num_feature, kernel_size=(1,1), strides= (1,1), padding='same', kernel_regularizer = regularizers.l2(1e-4))(x)
x = Dropout(0.2)(x)
x = AveragePooling2D((2,2), strides=(2,2), padding='same')(x)

## Dense block 2
x = Dense_block(x, num_layers=num_layers, growth_rate=growth_rate)

x = BatchNormalization(axis=3)(x)
num_feature = int(x.shape[3])
x = Conv2D(filters= num_feature, kernel_size=(1,1), strides= (1,1), padding='same', kernel_regularizer = regularizers.l2(1e-4))(x)
x = Dropout(0.2)(x)
x = AveragePooling2D((2,2), strides=(2,2), padding='same')(x)

## Dense block 3
x = Dense_block(x, num_layers=num_layers, growth_rate=growth_rate)

x = GlobalAveragePooling2D()(x)
x = Dense(10, activation='softmax')(x)

model = Model(inputs=inputs, outputs=x)

sgd = optimizers.SGD(lr=0.05, decay=0.0, momentum=0.9, nesterov=True)
#Adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-5)
#RMSprop = optimizers.RMSprop(lr=0.00005, rho=0.9, epsilon=1e-08, decay=1e-4)

model.compile(optimizer=sgd, loss='categorical_crossentropy',  metrics=['accuracy'])

#model.summary()

for i in xrange(0,30):
    model.fit(x_train, y_train, batch_size=32, epochs=5)
    score = model.evaluate(x_test, y_test, batch_size=32)
    
    print('  Test loss ',i,' :', score)
    

sgd = optimizers.SGD(lr=0.005, decay=0.0, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy',  metrics=['accuracy'])
for i in xrange(30,45):
    model.fit(x_train, y_train, batch_size=32, epochs=5)
    score = model.evaluate(x_test, y_test, batch_size=32)
    
    print('  Test loss ',i,' :', score)    

    
sgd = optimizers.SGD(lr=0.0005, decay=0.0, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy',  metrics=['accuracy'])
for i in xrange(45,60):
    model.fit(x_train, y_train, batch_size=32, epochs=5)
    score = model.evaluate(x_test, y_test, batch_size=32)
    
    print('  Test loss ',i,' :', score)    
