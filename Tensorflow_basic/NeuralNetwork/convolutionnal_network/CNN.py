#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by XiaoYu on 18-3-14
from __future__ import division, print_function, absolute_import

import tensorflow as tf

#导入MNIST数据
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./data", one_hot=True)

#训练参数
lr = 0.001
num_steps = 200
batch_size = 128
display_step = 10

#网络参数
num_input = 784
num_classes = 10
dropout = 0.75

#tf图输入
X = tf.placeholder(tf.float32, [None, num_input])
Y = tf.placeholder(tf.float32, [None, num_classes])
keep_prob = tf.placeholder(tf.float32)

#创建一些简单的包装
def conv2d(x, W, b, strides=1):
	x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
	x = tf.nn.bias_add(x, b)
	x = tf.nn.relu(x)
	return x

def maxpool2d(x, k=2):
	return(tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
	               padding='SAME'))

#模型构建
def conv_net(x, weights, biases, dropout):
	x = tf.reshape(x, shape=[-1, 28, 28, 1])
	
	#卷积层
	conv1 = conv2d(x, weights['wc1'], biases['bc1'])
	conv1 = maxpool2d(conv1, k=2)
	
	#卷积层
	conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
	conv2 = maxpool2d(conv2, k=2)
	
	#全连接层
	#转换conv2的输出去适应全连接输入
	fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
	fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
	fc1 = tf.nn.relu(fc1)
	#dropout
	fc1 = tf.nn.dropout(fc1, dropout)
	
	#输出，预测类别
	out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
	return out

#存储weights和biases
weights = {
	# 5*5卷积核 1 input 32 output
	'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
	# 5*5卷积核 32 input 64 output
	'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
	#全连接层， 7*7*64 input 1024 output
	'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])),
	#输出层 1024 input 10 output
	'out': tf.Variable(tf.random_normal([1024, 10]))
}

biases = {
	'bc1': tf.random_normal([32]),
	'bc2': tf.random_normal([64]),
	'bd1': tf.random_normal([1024]),
	'out': tf.random_normal([num_classes])
}

#创建模型
logits = conv_net(X, weights, biases, keep_prob)
prediction = tf.nn.softmax(logits)

#定义损失和优化
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
	logits=logits, labels=Y
))
optimizer = tf.train.AdamOptimizer(learning_rate=lr)
train_op = optimizer.minimize(loss_op)

#模型评估
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

#初始化所有变量
init = tf.global_variables_initializer()

#开始训练
with tf.Session() as sess:
	sess.run(init)
	
	for step in range(1, num_steps+1):
		batch_x, batch_y = mnist.train.next_batch(batch_size)
		#进行反向传播
		sess.run(train_op, feed_dict={X:batch_x, Y:batch_y, keep_prob:0.8})
		if step % display_step == 0 or step == 1:
			#计算误差和准确率
			loss, acc = sess.run([loss_op, accuracy],
			                     feed_dict={X:batch_x,
			                                Y:batch_y,
			                                keep_prob:1.0})
			print("Step " + str(step) + "最小批次误差 " + \
			      "{:.4f}".format(loss) + "训练准确率" + \
			      "{:.3f}".format(acc))
	
	print("优化完成")
	
	#计算测试率
	print("测试准确率 ", sess.run(accuracy, feed_dict={
		X: mnist.test.images[:256],
		Y: mnist.test.labels[:256],
		keep_prob: 1.0
	}))