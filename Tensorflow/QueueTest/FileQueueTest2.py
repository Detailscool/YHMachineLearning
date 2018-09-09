#!/usr/bin/python
# -*- coding:utf-8 -*-
#  FileQueueTest1.py
#  Created by HenryLee on 2018/9/9.
#  Copyright © 2018年. All rights reserved.
#  Description : Image

import tensorflow as tf
import os

file_list = [os.path.join('./tmp/data/dog', file) for file in os.listdir('./tmp/data/dog')]

file_queue = tf.train.string_input_producer(file_list)

reader = tf.WholeFileReader()

key, value = reader.read(file_queue)

image = tf.image.decode_jpeg(value)

image_resize = tf.image.resize_images(image, [200, 200])

image_resize.set_shape([200, 200, 3])

image_batch = tf.train.batch([image_resize], batch_size=20, num_threads=1, capacity=9)

with tf.Session() as sess:
    coord = tf.train.Coordinator()

    threads = tf.train.start_queue_runners(sess, coord=coord)

    print(sess.run([image_batch]))

    coord.request_stop()

    coord.join(threads)


