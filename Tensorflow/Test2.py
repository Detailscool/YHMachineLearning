#!/usr/bin/python
# -*- coding:utf-8 -*-
# Test2.py
# Created by Henry on 2018/9/5
# Description :

import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

g = tf.Graph()
print(g)

graph = tf.get_default_graph()
print(graph)

with g.as_default():
    c = tf.constant(3.0)
    print(c)
    print(c.graph)

a = tf.constant(11)
b = tf.constant(12)
sum = tf.add(a, b)
graph = tf.get_default_graph()
print(graph)

# with tf.Session(graph=g) as session:
#     print session.run(c)
#     print a.graph

# with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as session:
#     print sum.eval()
#     print session.run(a)
#     print a.graph

aa = tf.placeholder(tf.float32, [2, 3])

with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as session:
    print(session.run(aa, feed_dict={aa: [[1, 3, 2], [1, 4, 6]]}))

