# -*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np

# 创建数据
x_data = np.random.rand(100).astype(np.float32)
y_data = 0.1 * x_data + 0.3

# 搭建模型
w = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))

y = w * x_data + b

# 计算误差
loss = tf.reduce_mean(tf.square(y - y_data))

# 传播误差
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# 训练
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for step in range(201):
    sess.run(train)
    if not step % 20:
        print(step, sess.run(w), sess.run(b))
sess.close()
