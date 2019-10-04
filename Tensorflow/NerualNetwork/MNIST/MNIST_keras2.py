#!/usr/bin/python
# -*- coding:utf-8 -*-
#  MNIST_keras2.py
#  Created by HenryLee on 2019/2/24.
#  Copyright © 2018年. All rights reserved.
#  Description :

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils

batch_size = 128
nb_epoch = 10
nb_classes = 10
img_size = 28 * 28

(x_train, y_train), (x_test, y_test) = mnist.load_data()
X_train = x_train.reshape(y_train.shape[0], img_size).astype('float32')/255
X_test = x_test.reshape(y_test.shape[0], img_size).astype('float32')/255
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()
model.add(Dense(512, input_shape=(img_size,)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(512, input_shape=(512,)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(10, input_shape=(512,), activation='softmax'))
print(model.summary())

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test)
print('accuracy: {}'.format(score[1]))