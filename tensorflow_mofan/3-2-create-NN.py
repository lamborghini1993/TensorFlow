# -*- coding:utf-8 -*-

import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
os.environ["PATH"]="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin"


def liner_layer(inputs, in_size, out_size, activation_func=None):
    w = tf.Variable(tf.random_normal([in_size, out_size]))
    b = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    y = tf.matmul(inputs, w) + b
    if activation_func:
        y = activation_func(y)
    return y

# 构建所需的数据
x_data = np.linspace(-1, 1, 300)
x_data = x_data[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
noise = noise.astype(np.float32)
y_data = np.square(x_data) - 0.5 + noise

# 利用占位符定义我们所需的神经网络的输入。 
# tf.placeholder()就是代表占位符，这里的None代表无论输入有多少都可以
# 因为输入只有一个特征，所以这里是1
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

# 开始定义神经层了。输入层、隐藏层和输出层
hidder = liner_layer(xs, 1, 10, tf.nn.relu)
prediction = liner_layer(hidder, 10, 1)
# 定义误差函数
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
# 如何让机器学习提升它的准确率,以0.1的效率来最小化误差loss
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
init = tf.global_variables_initializer()

# fig = plt.figure()
# ax = fig.add_subplot(1,1,1)

with tf.Session() as sess:
    sess.run(init)
    # 开始学习
    for iStep in range(1001):
        sess.run(train_step, feed_dict={xs:x_data, ys:y_data})
        print(xs, ys)
        # if not iStep % 50:
        #     print(iStep, sess.run(loss, feed_dict={xs:x_data, ys:y_data}))
