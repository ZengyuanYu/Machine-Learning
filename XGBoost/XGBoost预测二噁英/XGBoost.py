#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by XiaoYu on 18-3-23
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
	train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.001, random_state=0)  ##test_size测试集合所占比例
	test_preds = pd.DataFrame({"label": test_y})
	clf = XGBRegressor(
		learning_rate=0.1,  # 默认0.3
		n_estimators=500,  # 树的个数
		max_depth=6,
		# min_child_weight=1,
		# gamma=0,
		# subsample=0.8,
		# colsample_bytree=0.8,
		# scale_pos_weight=1
	)
	clf.fit(train_x, train_y)
	test_preds['y_pred'] = clf.predict(test_x)
	stdm = metrics.r2_score(test_preds['label'], test_preds['y_pred'])
	a = cross_val_score(clf, X, y, cv=3)
	print(a)
	import matplotlib.pyplot as plt  # 画出预测结果图
	p = test_preds[['label', 'y_pred']].plot(subplots=True, style=['b-o', 'r-*'])
	plt.plot(test_preds['label'], c='blue')
	plt.plot(test_preds['y_pred'], c='red')
	plt.show()
	
	return stdm, clf


def GDBTest(X, y, clf):
	test_preds = pd.DataFrame({"label": y})
	test_preds['y_pred'] = clf.predict(X)
	print(len(test_preds))
	# # 对预测的结果取整
	# for i in range(16):
	# 	test_preds['y_pred'][i] = round(test_preds['y_pred'][i])
	# 	i += 1
	#
	error = 0.
	# print(test_preds)
	for i in range(16):
		a = abs(test_preds['y_pred'][i] - test_preds['label'][i])
		error += a / test_preds['label'][i]
		i += 1
	mean_error = error / 16
	print("平均误差为: %s" % mean_error)
	print("均方误差： ", metrics.mean_squared_error(
		test_preds['label'], test_preds['y_pred']))
	stdm = metrics.r2_score(test_preds['label'], test_preds['y_pred'])
	import matplotlib.pyplot as plt  # 画出预测结果图
	p = test_preds[['label', 'y_pred']].plot(subplots=True, style=['b-o', 'r-*'])
	plt.plot(test_preds['label'], c='blue')
	plt.plot(test_preds['y_pred'], c='red')
	plt.show()
	return stdm


def XGTSearch(X, y):
	print("Parameter optimization")
	n_estimators = [50, 100, 200, 400, 500, 600]
	max_depth = [2, 3, 4, 6, 8, 16]
	learning_rate = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]
	param_grid = dict(max_depth=max_depth, n_estimators=n_estimators, learning_rate=learning_rate)
	xgb_model = XGBRegressor()
	kfold = TimeSeriesSplit(n_splits=2).get_n_splits([X, y])
	fit_params = {"eval_metric": "rmse"}
	grid_search = GridSearchCV(xgb_model, param_grid, verbose=1, fit_params=fit_params, cv=kfold)
	grid_result = grid_search.fit(X, y)
	# summarize results
	print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
	means = grid_result.cv_results_['mean_test_score']
	stds = grid_result.cv_results_['std_test_score']
	params = grid_result.cv_results_['params']
	for mean, stdev, param in zip(means, stds, params):
		print("%f (%f) with: %r" % (mean, stdev, param))
	
	return means, grid_result


if __name__ == "__main__":
	X_data_train = pd.read_csv('X_train.csv')
	Y_data_train = pd.read_csv('Y_label.csv')
	Y_data_train = Y_data_train[u'二噁英排放浓度']
	out = 'frazee'
	print("输入数据类型：", X_data_train.shape)
	print("标签类型：", Y_data_train.shape)
	X = X_data_train.as_matrix()
	y = Y_data_train.as_matrix()
	stdm, clf = GDBTTrain(X, y)
	print('saving model')
	path = './model/' + out + '_xgb.pkl'
	with open(path, "wb") as f:
		pickle.dump(clf, f)
	print("训练r2_score结果", stdm)
# a,b = XGTSearch(X, y)
# 	X_data_test = pd.read_csv('X_16_feature.csv', encoding="gb18030")
# 	Y_data_test = pd.read_csv('Y_16.csv', encoding="gb18030")
# 	Y_data_test = Y_data_test['times']

# 	X = X_data_test.as_matrix()
# 	y = Y_data_test.as_matrix()
# 	path = './model/' + out + '_xgb.pkl'
# 	with open(path, 'rb') as f:
# 		model = pickle.load(f)
# 		stdm = GDBTest(X, y, model)
# 		print("测试r2_score结果：", stdm)
