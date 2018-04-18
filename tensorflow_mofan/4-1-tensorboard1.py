# -*- coding:utf-8 -*-
"""
@Author: lamborghini
@Date: 2018-04-18 14:12:30
@Desc: Tensorboard 可视化好帮手
"""
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def liner_layer(inputs, in_size, out_size, activation_func=None):
    w = tf.Variable(tf.random_normal([in_size, out_size]))
    b = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    y = tf.matmul(inputs, w) + b
    if activation_func:
        y = activation_func(y)
    return y


def add_layer(inputs, in_size, out_size, activation_func=None):
    """使用Tensorboard可视化"""
    with tf.name_scope("layer"):
        with tf.name_scope("weights"):
            w = tf.Variable(tf.random_normal([in_size, out_size]))
        with tf.name_scope("biases"):
            b = tf.Variable(tf.zeros([1, out_size]) + 0.1)
        with tf.name_scope("w_inputs_b"):
            y = tf.matmul(inputs, w) + b
        if activation_func:
            y = activation_func(y)
        return y


# Make up some real data
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

##plt.scatter(x_data, y_data)
##plt.show()

# define placeholder for inputs to network
with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, 1], name="x_input")
    ys = tf.placeholder(tf.float32, [None, 1], name="y_input")

# 开始定义神经层了。输入层、隐藏层和输出层
hidder = add_layer(xs, 1, 10, tf.nn.relu)
prediction = add_layer(hidder, 10, 1)
# 定义误差函数
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
# 如何让机器学习提升它的准确率,以0.1的效率来最小化误差loss
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
init = tf.global_variables_initializer()

# fig = plt.figure()
# ax = fig.add_subplot(1,1,1)

sess = tf.Session()
writer = tf.summary.FileWriter("logs/", sess.graph)
sess.run(init)

