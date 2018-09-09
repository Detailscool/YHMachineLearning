#!/usr/bin/python
# -*- coding:utf-8 -*-
#  QueueTest2.py
#  Created by HenryLee on 2018/9/9.
#  Copyright © 2018年. All rights reserved.
#  Description :

import tensorflow as tf

queue = tf.FIFOQueue(1000, tf.float32)

var = tf.Variable(0.0)

data = tf.assign_add(var, tf.constant(1.0))

enqueue = queue.enqueue(data)

qr = tf.train.QueueRunner(queue, enqueue_ops=[enqueue]*2)

init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)

    coord = tf.train.Coordinator()

    threads = qr.create_threads(sess, coord=coord, start=True)

    for i in range(10):
        print(sess.run(queue.dequeue()))

    coord.request_stop()
    coord.join(threads)