# coding:utf-8
# 实现Bi RNN

import tensorflow as tf
import tensorflow.contrib as tfc
import tensorflow.examples.tutorials.mnist.input_data as input_data
import numpy as np

# 读取MNIST数据集
mnist = input_data.read_data_sets('downloads/MNIST_data/', one_hot=True)

# 参数
learning_rate = 0.01
max_samples = 400000
batch_size = 128
display_step = 10

n_input = 28
n_steps = 28
n_hidden = 256
n_classes = 10

# 输入，n_steps是时间序列，这里用图片高度替代
x = tf.placeholder(tf.float32, [None, n_steps, n_input])
# label
y = tf.placeholder(tf.float32, [None, n_classes])

# Softmax层的权重，因为双向LSTM的输出有两个cell，所以参数量*2
weights = tf.Variable(tf.random_normal([2*n_hidden, n_classes]))
biases = tf.Variable(tf.random_normal([n_classes]))

# BiLSTM网络生成函数
def BiRNN(x, weight, biases):
    # 输入数据格式处理，
    # 原始：(batch_size, n_steps, n_input)，
    # 目标：长度为n_steps的列表，其中元素尺寸为(batch_size, n_input)
    # 将第一个维度batch_size与第二个维度n_steps交换
    x = tf.transpose(x, [1, 0, 2])
    # 变形为(n_steps*batch_size, n_input)
    x = tf.reshape(x, [-1, n_steps])
    # 切分为长度为n_steps的列表，其中元素尺寸为(batch_size, n_input)
    x = tf.split(x, n_steps)

    # 正向单元
    lstm_fw_cell = tfc.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
    # 反向单元
    lstm_bw_cell = tfc.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)

    # LSTM输出
    outputs, _, _ = tfc.rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x, dtype=tf.float32)

    # 乘上权重加上偏置作为输出
    return tf.matmul(outputs[-1], weights) + biases

pred = BiRNN(x, weights, biases)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    step = 1
    while step * batch_size < max_samples:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        batch_x = batch_x.reshape((batch_size, n_steps, n_input))
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        if step % display_step == 0:
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            print('Iter %d, Minibatch Loss=%.6f, Training Accuracy=%.5f' % (step*batch_size, loss, acc))
        step += 1

    print('Optimization Finished!')

    test_len = 10000
    test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
    test_label = mnist.test.labels[:test_len]
    print('Testing Accuracy: %.5f' % sess.run(accuracy, feed_dict={x: test_data, y: test_label}))