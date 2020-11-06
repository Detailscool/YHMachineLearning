#!/usr/bin/python
# -*- coding:utf-8 -*-
# Poetry_Generator.py
# Created by Henry on 2020/11/4
# Description :TensorFlow 1.13.1

import numpy as np
import pickle

# import sys
# import re
# pattern = '[^\u4e00-\u9fa5，。！：；？]+'
#
# text = open('./data/newtxt.txt', 'rb').read().decode(encoding='utf-8')
# vocab = sorted(set(re.sub(pattern, '', str(set(text)))))
# # vocab = sorted(set(set(text)))
# vocab_to_int = {u: i for i, u in enumerate (vocab)}
# print(vocab_to_int)
# int_to_vocab = np.array (vocab)
#
# int_text = np.array ([vocab_to_int[word] for word in text if vocab_to_int.get(word, '')!=''])
#
# pickle.dump ((int_text, vocab_to_int, int_to_vocab), open ('preprocess.p', 'wb'))
# sys.exit(0)



# 读取保存的数据
int_text, vocab_to_int, int_to_vocab = pickle.load(open('preprocess.p', mode='rb'))
print(vocab_to_int)


def get_batches(int_text, batch_size, seq_length):
    batchCnt = len(int_text) // (batch_size * seq_length)
    int_text_inputs = int_text[:batchCnt * (batch_size * seq_length)]
    int_text_targets = int_text[1:batchCnt * (batch_size * seq_length)+1]

    result_list = []
    x = np.array(int_text_inputs).reshape(1, batch_size, -1)
    y = np.array(int_text_targets).reshape(1, batch_size, -1)

    x_new = np.dsplit(x, batchCnt)
    y_new = np.dsplit(y, batchCnt)

    for ii in range(batchCnt):
        x_list = []
        x_list.append(x_new[ii][0])
        x_list.append(y_new[ii][0])
        result_list.append(x_list)

    return np.array(result_list)


def gen_poetry(prime_word='白', top_n=5, rule=7, sentence_lines=4, hidden_head=None):
    gen_length = sentence_lines * (rule + 1) - len (prime_word)
    gen_sentences = [prime_word] if hidden_head == None else [hidden_head[0]]
    temperature = 1.0

    dyn_input = np.array([vocab_to_int[s] for s in prime_word])
    dyn_input = np.expand_dims(dyn_input, 0)

    # dyn_seq_length = len (dyn_input[0])

    model.reset_states()
    index = len (prime_word) if hidden_head == None else 1
    for n in range (gen_length):
        index += 1

        if index != 0 and (index % (rule + 1)) == 0:
            if ((index / (rule + 1)) + 1) % 2 == 0:
                predicted_id = vocab_to_int['，']
            else:
                predicted_id = vocab_to_int['。']
        else:
            predictions = model(tf.constant(dyn_input))

            predictions = tf.squeeze (predictions, 0)

            a = predictions.eval ()

            if hidden_head != None and (index - 1) % (rule + 1) == 0 and (index - 1) // (rule + 1) < len (hidden_head):
                predicted_id = vocab_to_int[hidden_head[(index - 1) // (rule + 1)]]
            else:
                while True:
                    predictions = predictions / temperature
                    samples = tf.random.categorical(predictions, num_samples=1)
                    predicted_ids = samples.eval()
                    predicted_id = predicted_ids[-1, 0]

                    # p = np.squeeze(predictions[-1].numpy())
                    # p[np.argsort(p)[:-top_n]] = 0
                    # p = p / np.sum(p)
                    # c = np.random.choice(vocab_size, 1, p=p)[0]
                    # predicted_id=c
                    if (predicted_id != vocab_to_int['，'] and predicted_id != vocab_to_int['。']):
                        break
                        # using a multinomial distribution to predict the word returned by the model
                        #         predictions = predictions / temperature
                        #         predicted_id = tf.multinomial(predictions, num_samples=1)[-1,0].numpy()
        combineid = dyn_input[-1].tolist() + [predicted_id]
        dyn_input = np.expand_dims(combineid, 0)
        gen_sentences.append(int_to_vocab[predicted_id])

    poetry_script = ''.join(gen_sentences)
    poetry_script = poetry_script.replace('\n ', '\n')
    poetry_script = poetry_script.replace('( ', '(')

    return poetry_script


vocab_size = len(int_to_vocab)

# 批次大小
batch_size = 32  # 64
# RNN的大小（隐藏节点的维度）
rnn_size = 1000
# 嵌入层的维度
embed_dim = 256  # 这里做了调整，跟彩票预测的也不同了
# 序列的长度
seq_length = 15  # 注意这里已经不是1了，在古诗预测里面这个数值可以大一些，比如100也可以的

import os
import tensorflow as tf

train_batches = get_batches(int_text, batch_size, seq_length)
losses = {'train': [], 'test': []}

model_dir = "./model/"

x_true = tf.placeholder(tf.float32, [None, seq_length])
y_true = tf.placeholder (tf.float32, [None, seq_length])

model = tf.keras.Sequential([
            tf.keras.layers.Embedding(vocab_size, embed_dim, batch_input_shape=[batch_size, None]), # batch_input_shape 预测时改成1，训练时为batch_size
            tf.keras.layers.LSTM(rnn_size, return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform'),
            tf.keras.layers.Dense(vocab_size)
        ])

model.summary()

optimizer = tf.train.AdamOptimizer()

with tf.GradientTape() as tape:
    y_predict = model(x_true, training=True)
    loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_predict, from_logits=True)

grads = tape.gradient(loss, model.trainable_variables)

optimizer.apply_gradients(zip(grads, model.trainable_variables))

train_op = optimizer.minimize(tf.reduce_sum(loss))

init_op = tf.global_variables_initializer()

tf.summary.scalar('loss', tf.reduce_mean(loss))

merged = tf.summary.merge_all()

saver = tf.train.Saver(max_to_keep=2)

with tf.Session() as sess:
    sess.run(init_op)

    if os.path.exists(model_dir):
        model_file = tf.train.latest_checkpoint(model_dir)
        saver.restore(sess, model_file)

        print (gen_poetry (prime_word='白', top_n=10, rule=5, sentence_lines=4, hidden_head='小泉红叶'))
    else:
        file_writer = tf.summary.FileWriter('./tmp/summary_conv', graph=sess.graph)

        epochs = 20
        for epoch in range(epochs):
            avg_cost = 0

            for batch_i, (x, y) in enumerate (train_batches):
                # print ("x_shape: %s y_true_shape: %s" % (_x.shape, y.shape))
                feed_dict = {x_true: x, y_true: y}

                sess.run(train_op, feed_dict=feed_dict)

                avg_loss = sess.run(tf.reduce_mean(loss), feed_dict=feed_dict)

                losses['train'].append(avg_loss)

                avg_cost += avg_loss / len(train_batches)

                summary = sess.run(merged, feed_dict=feed_dict)

                file_writer.add_summary(summary, batch_i)

            saver.save(sess, model_dir+'poetry.cpkt', global_step=epoch)

            print("Epoch: {} avg_cost:{}".format(epoch, avg_cost))

        import matplotlib.pyplot as plt

        plt.plot(losses['train'], label='Training loss')
        plt.show()