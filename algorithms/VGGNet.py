from keras.datasets import mnist, cifar10
(img_train, label_train), (img_test, label_test) = cifar10.load_data()
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras import losses, optimizers

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

init_size = 2
model = Sequential()

model.add(Conv2D(filters=init_size, kernel_size=(3, 3), padding='same', input_shape = (row, col, depth)))
model.add(BatchNormalization(axis=3))
model.add(Activation('relu'))
model.add(Conv2D(init_size, (3, 3), padding='same'))
model.add(BatchNormalization(axis=3))
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2), padding='same'))

model.add(Conv2D(init_size*2, (3, 3), padding='same'))
model.add(BatchNormalization(axis=3))
model.add(Activation('relu'))
model.add(Conv2D(init_size*2, (3, 3), padding='same'))
model.add(BatchNormalization(axis=3))
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2), padding='same'))

model.add(Conv2D(init_size*4, (3, 3), padding='same'))
model.add(BatchNormalization(axis=3))
model.add(Activation('relu'))
model.add(Conv2D(init_size*4, (3, 3), padding='same'))
model.add(BatchNormalization(axis=3))
model.add(Activation('relu'))
model.add(Conv2D(init_size*4, (3, 3), padding='same'))
model.add(BatchNormalization(axis=3))
model.add(Activation('relu'))
model.add(Conv2D(init_size*4, (3, 3), padding='same'))
model.add(BatchNormalization(axis=3))
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2), padding='same'))

"""model.add(Conv2D(init_size*8, (3, 3), padding='same'))
model.add(BatchNormalization(axis=3))
model.add(Activation('relu'))
model.add(Conv2D(init_size*8, (3, 3), padding='same'))
model.add(BatchNormalization(axis=3))
model.add(Activation('relu'))
model.add(Conv2D(init_size*8, (3, 3), padding='same'))
model.add(BatchNormalization(axis=3))
model.add(Activation('relu'))
model.add(Conv2D(init_size*8, (3, 3), padding='same'))
model.add(BatchNormalization(axis=3))
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2), padding='same'))

model.add(Conv2D(init_size*8, (3, 3), padding='same'))
model.add(BatchNormalization(axis=3))
model.add(Activation('relu'))
model.add(Conv2D(init_size*8, (3, 3), padding='same'))
model.add(BatchNormalization(axis=3))
model.add(Activation('relu'))
model.add(Conv2D(init_size*8, (3, 3), padding='same'))
model.add(BatchNormalization(axis=3))
model.add(Activation('relu'))
model.add(Conv2D(init_size*8, (3, 3), padding='same'))
model.add(BatchNormalization(axis=3))
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2), padding='same'))"""

model.add(Flatten())
model.add(Dense(init_size*64))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
#model.add(Dropout(0.5))
model.add(Dense(init_size*64))
model.add(BatchNormalization(axis=-1))
#model.add(Dropout(0.5))
model.add(Dense(label_train.shape[1], activation='softmax'))

model.summary()

model.compile(loss=losses.categorical_crossentropy, 
              optimizer=optimizers.Adam(lr=0.01),
              metrics=['accuracy'])
model.fit(img_train, label_train, batch_size=256, epochs=1000, validation_split=0.1)
score = model.evaluate(img_test, label_test)

print "Test score : ", score[0]
print "Test accuracy : ", score[1]