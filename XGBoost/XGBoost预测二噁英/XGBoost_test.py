#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by XiaoYu on 18-3-28
import pandas as pd
# import numpy as np
from sklearn import metrics
import pickle
from xgboost.sklearn import XGBRegressor
# from sklearn.preprocessing import StandardScaler
# from clean_data import prep_water_data,normalize_water_data,normalize_data,delete_null_date
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.model_selection import TimeSeriesSplit


def GDBTTrain(X, y):
	train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=0)  ##test_size测试集合所占比例
	test_preds = pd.DataFrame({"label": test_y})
	clf = XGBRegressor(
		learning_rate=0.1,  # 默认0.3
		n_estimators=400,  # 树的个数
		max_depth=6,
	)
	clf.fit(train_x, train_y)
	test_preds['y_pred'] = clf.predict(test_x)
	stdm = metrics.r2_score(test_preds['label'], test_preds['y_pred'])
	print(test_preds[0:10])

	return stdm, clf

def GDBTest(X, y, clf):
	test_preds = pd.DataFrame({"label": y})
	test_preds['y_pred'] = clf.predict(X)
	print("验证集的个数", len(test_preds))
	stdm = metrics.r2_score(test_preds['label'], test_preds['y_pred'])
	loss = -cross_val_score(clf, X, y, cv=10, scoring='neg_mean_squared_error')
	print("交叉验证平方差", loss.mean())
	return stdm, test_preds

if __name__ == "__main__":
	#1.训练
	#读取训练集的数据并转换
	X_data_train = pd.read_csv('X_train_split.csv')
	Y_data_train = pd.read_csv('Y_train_split.csv')
	Y_data_train = Y_data_train[u'二噁英排放浓度']
	out = 'frazee'
	print("输入数据类型：", X_data_train.shape)
	print("标签类型：", Y_data_train.shape)
	X = X_data_train.as_matrix()
	y = Y_data_train.as_matrix()
	#训练和测试准确度
	stdm, clf = GDBTTrain(X, y)
	#存储模型
	print('saving model')
	path = './model/' + out + '_xgb.pkl'
	with open(path, "wb") as f:
		pickle.dump(clf, f)
	print("训练r2_score结果", stdm)
	
	#2.验证
	#读取验证集数据
	X_data_test = pd.read_csv('X_val_split_clear.csv')[:]
	Y_data_test = pd.read_csv('Y_val_split_clear.csv')[:]
	Y_data_test = Y_data_test[u'二噁英排放浓度']

	print("输入数据类型：", X_data_test.shape)
	print("标签类型：", Y_data_test.shape)
	X = X_data_test.as_matrix()
	y = Y_data_test.as_matrix()
	
	#模型载入并测试
	with open(path, 'rb') as f:
		clf = pickle.load(f)
		stdm, prediction = GDBTest(X, y, clf)
	print("验证r2_score的值：", stdm)
	print("输出验证结果：", prediction)
	
	#3.可视化
	# 绘制实际结果和预测结果
	import matplotlib.pyplot as plt
	import numpy as np
	
	fig = plt.figure(figsize=(15, 6))
	plt.subplot(1, 2, 1)
	x = np.arange(len(prediction))
	plt.scatter(x, prediction['y_pred'])
	plt.scatter(x, prediction['label'], c='r')
	plt.xlabel('number')
	plt.ylabel('value')
	plt.ylim(0, 0.02)
	plt.grid(True)
	plt.legend(labels=['pre', 'label'], loc='best')
	plt.show()
