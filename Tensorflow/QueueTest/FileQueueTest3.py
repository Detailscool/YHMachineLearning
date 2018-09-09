#!/usr/bin/python
# -*- coding:utf-8 -*-
#  FileQueueTest1.py
#  Created by HenryLee on 2018/9/9.
#  Copyright © 2018年. All rights reserved.
#  Description : Bytes

import tensorflow as tf
import os

width = 32
height = 32
channel = 3
label_bytes = 1
bytes = width * height * channel + label_bytes

file_list = [os.path.join('./tmp/data/cifar-10-batches-bin', file) for file in os.listdir('./tmp/data/cifar-10-batches-bin') if file.endswith('.bin')]

file_queue = tf.train.string_input_producer(file_list)

reader = tf.FixedLengthRecordReader(bytes)

key, value = reader.read(file_queue)

label_image = tf.decode_raw(value, tf.uint8)

label = tf.cast(tf.slice(label_image, [0], [label_bytes]), tf.int32)

print(label)

image = tf.slice(label_image, [label_bytes], [bytes-label_bytes])

print(image)

image_reshape = tf.reshape(image, [height, width, channel])

label_batch, image_batch = tf.train.batch([label, image_reshape], batch_size=10, num_threads=1, capacity=10)

with tf.Session() as sess:
    coord = tf.train.Coordinator()

    threads = tf.train.start_queue_runners(sess, coord=coord)

    print(sess.run([label_batch, image_batch]))

    coord.request_stop()

    coord.join(threads)


