#!/usr/bin/python
# -*- coding:utf-8 -*-
#  MNIST_keras.py
#  Created by HenryLee on 2019/2/24.
#  Copyright © 2018年. All rights reserved.
#  Description :

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils

batch_size = 128
nb_classes = 10
nb_epoch = 10
img_size = 28 * 28

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# print(x_train.shape, x_test.shape)
# print(y_train.shape, y_test.shape)
X_train = x_train.reshape(y_train.shape[0], img_size).astype('float32')/255
X_test = x_test.reshape(y_test.shape[0], img_size).astype('float32')/255
#
print(X_train.shape, X_test.shape)

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()
model.add(Dense(128, input_shape=(img_size,), activation='relu'))
model.add(Dense(10, input_shape=(128,), activation='softmax'))
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_data=(X_test, Y_test))

score = model.evaluate(X_test, Y_test)
print('accuracy: {}'.format(score[1]))
