#!/usr/bin/python
# -*- coding:utf-8 -*-
#  MNIST_RNN.py
#  Created by HenryLee on 2019/10/5.
#  Copyright © 2018年. All rights reserved.
#  Description :

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('./tmp/MNIST_data', one_hot=True)

dim_input = 28
dim_hidden = 128
dim_output = mnist.train.labels.shape[1]
n_steps = 28

x = tf.placeholder(tf.float32, [None, n_steps, dim_input])
y = tf.placeholder(tf.float32, [None, dim_output])

_X = tf.transpose(x, [1, 0, 2])
_X = tf.reshape(_X, [-1, dim_input])
weight1 = tf.Variable(tf.random_normal([dim_input, dim_hidden]), dtype=tf.float32)
bias1 = tf.Variable(tf.random_normal([dim_hidden]), dtype=tf.float32)
_H = tf.add(tf.matmul(_X, weight1), bias1)
_Hsplit = tf.split(_H, n_steps, 0)

with tf.variable_scope('basic', reuse=None) as scope:
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(dim_hidden, forget_bias=1.0)
    _LSTM_O, _LSTM_S = tf.contrib.rnn.static_rnn(lstm_cell, _Hsplit, dtype=tf.float32)
weight2 = tf.Variable(tf.random_normal([dim_hidden, dim_output]))
bias2 = tf.Variable(tf.random_normal([dim_output]))
y_pred = tf.add(tf.matmul(_LSTM_O[-1], weight2), bias2)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_pred))
opt = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1)), tf.float32))

init_op = tf.global_variables_initializer()

batch_size = 16
test_img = mnist.test.images
test_labels = mnist.test.labels
n_tests = test_img.shape[0]
test_imgs = test_img.reshape((n_tests, n_steps, dim_input))

with tf.Session() as sess:
    sess.run(init_op)

    for epoch in range(5):
        avg_cost = 0
        total_batch = 100

        for i in range(total_batch):
            mnist_x, mnist_y = mnist.train.next_batch(batch_size)

            mnist_xs = mnist_x.reshape((batch_size, n_steps, dim_input))

            feed_dict = {x: mnist_xs, y: mnist_y}

            sess.run(opt, feed_dict=feed_dict)

            avg_cost += sess.run(loss, feed_dict=feed_dict)/total_batch

        print("Epoch: {}".format(epoch),
              "Training Acc: {}".format(sess.run(acc, feed_dict=feed_dict)),
              "Test Acc: {}".format(sess.run(acc, feed_dict={x: test_imgs, y: test_labels})))


