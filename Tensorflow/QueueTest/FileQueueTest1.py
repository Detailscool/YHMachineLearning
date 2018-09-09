#!/usr/bin/python
# -*- coding:utf-8 -*-
#  FileQueueTest1.py
#  Created by HenryLee on 2018/9/9.
#  Copyright © 2018年. All rights reserved.
#  Description : CSV

import tensorflow as tf
import os

file_list = [os.path.join('./tmp/csv', file) for file in os.listdir('./tmp/csv')]

file_queue = tf.train.string_input_producer(file_list)

reader = tf.TextLineReader()

key, value = reader.read(file_queue)

example, label = tf.decode_csv(value, record_defaults=[['None'], ['None']])

example_batch, label_batch = tf.train.batch([example, label], batch_size=20, num_threads=1, capacity=9)

with tf.Session() as sess:
    coord = tf.train.Coordinator()

    threads = tf.train.start_queue_runners(sess, coord=coord)

    print(sess.run(key))

    print(sess.run([example_batch, label_batch]))

    coord.request_stop()

    coord.join(threads)


