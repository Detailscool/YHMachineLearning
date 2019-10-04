#!/usr/bin/python
# -*- coding:utf-8 -*-
#  MNIST_GAN.py
#  Created by HenryLee on 2019/10/4.
#  Copyright © 2018年. All rights reserved.
#  Description :

import tensorflow as tf


# 生成器
def get_generator(noise_img, n_units, out_dim, reuse=False, alpha=0.01):
    with tf.variable_scope("generator", reuse=reuse):
        hiddenl = tf.layers.dense(noise_img, n_units)
        hiddenl = tf.maximum(alpha * hiddenl, hiddenl)
        hiddenl = tf.layers.dropout(hiddenl, rate=0.2)

        logits = tf.layers.dense(hiddenl, out_dim)
        outputs = tf.tanh(logits)

        return logits, outputs


# 判别器
def get_discriminator(img, n_units, reuse=False, alpha=0.01):
    with tf.variable_scope("discriminator", reuse=reuse):
        hiddenl = tf.layers.dense(img, n_units)
        hiddenl = tf.maximum(alpha * hiddenl, hiddenl)

        logits = tf.layers.dense(hiddenl, 1)
        outputs = tf.sigmoid(logits)

        return logits, outputs


from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('./tmp/MNIST_data', one_hot=True)

img_size = mnist.train.images[0].shape[0]

noice_size = 100

g_units = 128

d_units = 128

learning_rate = 0.001

alpha = 0.01

# 构建网络
tf.reset_default_graph()

real_img = tf.placeholder(tf.float32, [None, img_size])
noise_img = tf.placeholder(tf.float32, [None, noice_size])

g_logists, g_outputs = get_generator(noise_img, g_units, img_size)

d_logits_real, d_outputs_real = get_discriminator(real_img, d_units)
d_logits_fake, d_outputs_fake = get_discriminator(g_outputs, d_units, reuse=True)


# 判别器的loss
d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real, labels=tf.ones_like(d_logits_real)))
# 识别生成的图片
d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=tf.zeros_like(d_logits_fake)))
# 总体的loss
d_loss = tf.add(d_loss_real, d_loss_fake)

g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=tf.ones_like(d_logits_fake)))

# 优化器
train_vars = tf.trainable_variables()

g_vars = [var for var in train_vars if var.name.startswith("generator")]

d_vars = [var for var in train_vars if var.name.startswith("discriminator")]

d_train_opt = tf.train.AdamOptimizer(learning_rate).minimize(d_loss, var_list=d_vars)
g_train_opt = tf.train.AdamOptimizer(learning_rate).minimize(g_loss, var_list=g_vars)


# 训练

batch_size = 64

epochs = 300

n_sample = 25

samples = []

losses = []

saver = tf.train.Saver(var_list=g_vars)

import numpy as np

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for e in range(epochs):
        for batch_i in range(mnist.train.num_examples//batch_size):
            batch = mnist.train.next_batch(batch_size)

            batch_images = batch[0].reshape(batch_size, 784)

            batch_images = batch_images * 2 -1

            batch_noise = np.random.uniform(-1, 1, size=(batch_size, noice_size))

            _ = sess.run(d_train_opt, feed_dict={real_img: batch_images, noise_img: batch_noise})

            _ = sess.run(g_train_opt, feed_dict={noise_img: batch_noise})

        # 每一轮结束计算loss
        train_loss_d = sess.run(d_loss, feed_dict={real_img: batch_images, noise_img: batch_noise})

        train_loss_d_real = sess.run(d_loss_real, feed_dict={real_img: batch_images, noise_img: batch_noise})

        train_loss_d_fake = sess.run(d_loss_fake, feed_dict={real_img: batch_images, noise_img: batch_noise})

        train_loss_g = sess.run(g_loss, feed_dict={noise_img: batch_noise})

        print("Epoch{}/{}...".format(e+1, epochs),
              "判别器损失: {:.4f}(判别器真实的：{:.4f} + 判别生成的: {:.4f})...".format(train_loss_d, train_loss_d_real, train_loss_d_fake),
              "生成器损失: {:.4f}".format(train_loss_g))

        losses.append((train_loss_d, train_loss_d_real, train_loss_d_fake, train_loss_g))

        sample_noise = np.random.uniform(-1, 1, size=(n_sample, noice_size))

        gen_samples = sess.run(get_generator(noise_img, g_units, img_size, reuse=True), feed_dict={noise_img: sample_noise})

        samples.append(gen_samples)

        saver.save(sess, './tmp/GAN/generator.ckpt')

