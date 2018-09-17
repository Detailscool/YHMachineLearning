#!/usr/bin/python
# -*- coding:utf-8 -*-
#  MNIST_CONV.py
#  Created by HenryLee on 2018/9/17.
#  Copyright © 2018年. All rights reserved.
#  Description :

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('./tmp/MNIST_data', one_hot=True)

with tf.variable_scope('data'):
    x = tf.placeholder(tf.float32, [None, 784])
    y_true = tf.placeholder(tf.float32, [None, 10])

with tf.variable_scope('conv1'):
    weight1 = tf.Variable(tf.random_normal([5, 5, 1, 32], mean=0, stddev=1.0))
    bias1 = tf.constant(0.0, shape=[32])

    x_input = tf.reshape(x, [-1, 28, 28, 1])

    # [None, 28, 28, 1] --> [None, 28, 28, 32]
    x_relu1 = tf.nn.relu(tf.nn.conv2d(x_input, weight1, strides=[1, 1, 1, 1], padding='SAME') + bias1)
    # [None, 28, 28, 32] --> [None, 14, 14, 32]
    x_maxpool1 = tf.nn.max_pool(x_relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

with tf.variable_scope('conv2'):
    weight2 = tf.Variable(tf.random_normal([5, 5, 32, 64], mean=0, stddev=1.0))
    bias2 = tf.constant(0.0, shape=[64])

    # [None, 14, 14, 32] --> [None, 14, 14, 64]
    x_relu2 = tf.nn.relu(tf.nn.conv2d(x_maxpool1, weight2, strides=[1, 1, 1, 1], padding='SAME') + bias2)
    # [None, 14, 14, 64] --> [None, 7, 7, 64]
    x_maxpool2 = tf.nn.max_pool(x_relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

with tf.variable_scope('full_connect'):
    weight_fc = tf.Variable(tf.random_normal([7*7*64, 10], mean=0, stddev=1.0))
    bias_fc = tf.constant(0.0, shape=[10])
    # [None, 7, 7, 64] --> [None, 7*7*64]
    x_fc = tf.reshape(x_maxpool2, [-1, 7*7*64])
    y_predict = tf.matmul(x_fc, weight_fc) + bias_fc

with tf.variable_scope('soft_cross'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_predict))

with tf.variable_scope('optimizer'):
    train_op = tf.train.GradientDescentOptimizer(0.0001).minimize(loss)

with tf.variable_scope('acc'):
    equal_list = tf.equal(tf.argmax(y_true, 1), tf.argmax(y_predict, 1))
    accuracy = tf.reduce_mean(tf.cast(equal_list, tf.float32))

tf.summary.scalar('loss', loss)
tf.summary.scalar('accuracy', accuracy)

tf.global_variables_initializer()

init_op = tf.global_variables_initializer()

merged = tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(init_op)

    file_writer = tf.summary.FileWriter('./tmp/summary_conv', graph=sess.graph)

    for i in range(2000):
        mnist_x, mnist_y = mnist.train.next_batch(50)

        feed_dict = {x: mnist_x, y_true: mnist_y}

        sess.run(train_op, feed_dict=feed_dict)

        print('第%i次，accuracy:%s' % (i, sess.run(accuracy, feed_dict=feed_dict)))

        summary = sess.run(merged, feed_dict=feed_dict)

        file_writer.add_summary(summary, i)

