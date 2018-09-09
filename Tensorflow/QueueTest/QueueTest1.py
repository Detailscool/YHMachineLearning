#!/usr/bin/python
# -*- coding:utf-8 -*-
#  QueueTest1.py
#  Created by HenryLee on 2018/9/8.
#  Copyright © 2018年. All rights reserved.
#  Description :

import tensorflow as tf

queue = tf.FIFOQueue(3, tf.float32)

enqueue_many = queue.enqueue_many(([0.1, 0.2, 0.3], ))

out_queue = queue.dequeue()

data = out_queue + 1

enqueue = queue.enqueue(data)

with tf.Session() as sess:
    sess.run(enqueue_many)

    for i in range(100):
        sess.run(enqueue)

    for i in range(queue.size().eval()):
        print(sess.run(out_queue))
