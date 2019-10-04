#!/usr/bin/python
# -*- coding:utf-8 -*-
# Test3.py
# Created by Henry on 2018/9/6
# Description :

import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

a = tf.constant([1, 2, 3, 4, 5])

var = tf.Variable(tf.random_normal([2, 3], mean=0, stddev=1.0))

print(a)
print(var)

init_op = tf.global_variables_initializer()

with tf.Session() as session:
    print(session.run([a, var]))