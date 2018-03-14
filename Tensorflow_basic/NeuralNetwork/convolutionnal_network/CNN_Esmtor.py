#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by XiaoYu on 18-3-14
from __future__ import division, print_function, absolute_import

#导入MNIST数据
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./data", one_hot=False)

import tensorflow as tf

#训练参数
learning_rate = 0.001
num_step = 2000
batch_size = 128

#网络参数
num_input = 784
num_classes = 10
dropout = 0.25

#构建神经网络
def conv_net(x_dict, n_classes, dropout, reuse, is_training):
	with tf.variable_scope('ConvNet', reuse=reuse):
		x = x_dict['images']
		
		x = tf.reshape(x, shape=[-1, 28, 28, 1])
		
		#32个滤波器 5*5卷积核
		conv1 = tf.layers.conv2d(x, 32, 5, activation=tf.nn.relu)
		conv1 = tf.layers.max_pooling2d(conv1, 2, 2)
		
		#64个滤波器 3*3卷积核
		conv2 = tf.layers.conv2d(conv1, 64, 3, activation=tf.nn.relu)
		conv2 = tf.layers.max_pooling2d(conv2, 2, 2)
		
		#将高维向量降低为一维
		fc1 = tf.contrib.layers.flatten(conv2)
		
		#全连接层
		fc1 = tf.layers.dense(fc1, 1024)
		#dropout
		fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)
		
		#输出
		out = tf.layers.dense(fc1, n_classes)
	return out

#定义模型函数
def model_fn(features, labels, mode):
	logitis_train = conv_net(features, num_classes, dropout,
	                         reuse=False, is_training=True)
	logitis_test = conv_net(features, num_classes, dropout,
	                        reuse=True, is_training=False)
	
	#预测
	pred_classes = tf.argmax(logitis_test, axis=1)
	pred_probas = tf.nn.softmax(logitis_test)
	
	if mode == tf.estimator.ModeKeys.PREDICT:
		return tf.estimator.EstimatorSpec(mode, predictions=pred_classes)
	
	#误差和优化
	loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
		logits=logitis_train, labels=tf.cast(labels, dtype=tf.int32)
	))
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
	train_op = optimizer.minimize(loss_op,
	                              global_step=tf.train.get_global_step())
	
	#评估模型的准确率
	acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)
	
	estim_specs = tf.estimator.EstimatorSpec(
		mode = mode,
		predictions=pred_classes,
		loss = loss_op,
		train_op=train_op,
		eval_metric_ops={'accuracy': acc_op}
	)
	return estim_specs

#创建评估
model = tf.estimator.Estimator(model_fn)

#定义输入
input_fn = tf.estimator.inputs.numpy_input_fn(
	x={'images':mnist.train.images}, y=mnist.train.labels,
	batch_size=batch_size, num_epochs=None, shuffle=True
)

#训练模型
model.train(input_fn, steps=num_step)

#评估模型
#定义评估的输入
input_fn = tf.estimator.inputs.numpy_input_fn(
	x={'images':mnist.test.images}, y=mnist.test.labels,
	batch_size=batch_size, shuffle=False
)

e = model.evaluate(input_fn)

print("测试准确率 %s"%e['accuracy'])