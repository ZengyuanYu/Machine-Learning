#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by XiaoYu on 18-3-13
import tensorflow as tf
import numpy as np

# 创建数据
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3

# 创建tensorflow结构
Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0)) #1维 数值区间在（-1~1）之间
biases = tf.Variable(tf.zeros([1]))

y = Weights * x_data + biases

loss = tf.reduce_mean(tf.square(y-y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()
# 创建tensorflow结构

sess = tf.Session()
sess.run(init)

for step in range(201):
	sess.run(train)
	if step%20 == 0:
		print(step, sess.run(Weights), sess.run(biases))