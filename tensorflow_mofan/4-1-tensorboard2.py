# -*- coding:utf-8 -*-
"""
@Author: lamborghini
@Date: 2018-04-18 14:12:30
@Desc: Tensorboard 可视化好帮手
"""
import os
import tensorflow as tf
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def add_layer(inputs, in_size, out_size, n_layer, activation_func=None):
    """使用Tensorboard可视化"""
    layer_name = "layer_%s" % n_layer
    with tf.name_scope("layer"):
        with tf.name_scope("weights"):
            w = tf.Variable(tf.random_normal([in_size, out_size]))
            tf.summary.histogram(layer_name + "/weights", w)
        with tf.name_scope("biases"):
            b = tf.Variable(tf.zeros([1, out_size]) + 0.1)
            tf.summary.histogram(layer_name + "/biases", b)
        with tf.name_scope("w_inputs_b"):
            y = tf.matmul(inputs, w) + b
        tf.summary.histogram(layer_name + "/output", y)        
        if activation_func:
            y = activation_func(y)
        tf.summary.histogram(layer_name + "/output_activation", y)
        return y


# Make up some real data
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise


# define placeholder for inputs to network
with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, 1], name="x_input")
    ys = tf.placeholder(tf.float32, [None, 1], name="y_input")

# 开始定义神经层了。输入层、隐藏层和输出层
hidder = add_layer(xs, 1, 10, "hidder", tf.nn.relu)
prediction = add_layer(hidder, 10, 1, "prediction")

# 定义误差函数
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
    tf.summary.scalar("loss", loss)

# 如何让机器学习提升它的准确率,以0.1的效率来最小化误差loss
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
    # tf.summary.scalar("train_step", train_step)

merged = tf.summary.merge_all()
sess = tf.Session()
init = tf.global_variables_initializer()
writer = tf.summary.FileWriter("logs/", sess.graph)
sess.run(init)

for i in range(1001):
    sess.run(train_step, feed_dict={xs:x_data,ys:y_data})
    # if i % 50:
    result = sess.run(merged, feed_dict={xs:x_data,ys:y_data})
    writer.add_summary(result, i)
