#!/usr/bin/python
# -*- coding:utf-8 -*-
#  MNIST.py
#  Created by HenryLee on 2018/9/16.
#  Copyright © 2018年. All rights reserved.
#  Description :

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('./tmp/MNIST_data', one_hot=True)

with tf.variable_scope('data'):
    x = tf.placeholder(tf.float32, [None, 784])
    y_true = tf.placeholder(tf.float32, [None, 10])

with tf.variable_scope('fc_model'):
    weight = tf.Variable(tf.random_normal([784, 10], mean=0.0, stddev=1.0), name='Weight')
    bias = tf.Variable(tf.constant(0.0, shape=[10]))

    y_predict = tf.matmul(x, weight) + bias

with tf.variable_scope('soft_cross'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_predict))

with tf.variable_scope('optimizer'):
    train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.variable_scope('acc'):
    equal_list = tf.equal(tf.argmax(y_true, 1), tf.argmax(y_predict, 1))
    accuracy = tf.reduce_mean(tf.cast(equal_list, tf.float32))

tf.summary.scalar('loss', loss)
tf.summary.scalar('accuracy', accuracy)

tf.summary.histogram('weight', weight)
tf.summary.histogram('bias', bias)

init_op = tf.global_variables_initializer()

merged = tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(init_op)

    file_writer = tf.summary.FileWriter('./tmp/summary', graph=sess.graph)

    for i in range(2000):
        mnist_x, mnist_y = mnist.train.next_batch(50)

        feed_dict = {x: mnist_x, y_true: mnist_y}

        sess.run(train_op, feed_dict=feed_dict)

        print('第%i次，accuracy:%s' % (i, sess.run(accuracy, feed_dict=feed_dict)))

        summary = sess.run(merged, feed_dict=feed_dict)

        file_writer.add_summary(summary, i)
