#!/usr/bin/python
# -*- coding:utf-8 -*-
#  LinerTest.py
#  Created by HenryLee on 2018/9/6.
#  Copyright © 2018年. All rights reserved.
#  Description :

import tensorflow as tf

x = tf.random_normal([100, 1], mean=2.98, stddev=3.1, name='x_data')

y_true = tf.matmul(x, [[0.8]]) + 1.1

weight = tf.Variable(tf.random_normal([1, 1], mean=1, stddev=0.5), name='weight')

bias = tf.Variable(0.0, name='bias')

y_predict = tf.matmul(x, weight) + bias

loss = tf.reduce_mean(tf.square(y_true - y_predict))

gradient_op = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

init_op = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(init_op)

    file_writer = tf.summary.FileWriter('./tmp/summary', graph=session.graph)

    print("第0次，权重:%f, 偏置:%f" % (weight.eval(), bias.eval()))
    for i in range(200):
        session.run(gradient_op)
        print("第%d次，权重:%f, 偏置:%f" % (i+1, weight.eval(), bias.eval()))