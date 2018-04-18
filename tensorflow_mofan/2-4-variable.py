# -*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np

state = tf.Variable(0, name="counter")
# print(state)

# 定义常量 10
ten = tf.constant(10)

# 定义加法步骤 (注: 此步并没有直接计算)
new_value = tf.add(state, ten)
# 将 State 更新成 new_value
update = tf.assign(state, new_value)

# 如果你在 Tensorflow 中设定了变量，那么初始化变量是最重要的！！所以定义了变量以后
# 一定要定义 init = tf.initialize_all_variables()
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)  #激活init
    for _ in range(10):
        sess.run(update)
        print(sess.run(state))
