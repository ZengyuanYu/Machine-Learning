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

def GDBTest(X, y, clf):
    test_preds = pd.DataFrame({"label": y})
    test_preds['y_pred'] = clf.predict(X)
    print("验证集的个数", len(test_preds))
    stdm = metrics.r2_score(test_preds['label'], test_preds['y_pred'])
    loss = -cross_val_score(clf, X, y, cv=10, scoring='neg_mean_squared_error')
    print("交叉验证平方差", loss.mean())
    cross_val = cross_val_score(clf, X, y, cv=10)
    print("交叉验证的结果： ",cross_val)    
    return stdm, test_preds

if __name__ == "__main__":

    #验证
    #读取验证集数据
    X_data_test = pd.read_csv('X_data.csv')[0:1000]
    Y_data_test = pd.read_csv('Y_label.csv')[0:1000]
    Y_data_test = Y_data_test[u'二噁英排放浓度']

    print("输入数据类型：", X_data_test.shape)
    print("标签类型：", Y_data_test.shape)
    X = X_data_test.as_matrix()
    y = Y_data_test.as_matrix()
    out = 'frazee'
    path = './model/' + out + '_xgb.pkl'
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


    x = np.arange(len(prediction))
    plt.scatter(x, prediction['y_pred'])
    plt.scatter(x, prediction['label'], c='r')
    plt.xlabel('number')
    plt.ylabel('value')
#     plt.ylim(0, 0.02)
    plt.grid(True)
    plt.legend(labels=['pre', 'label'], loc='best')
    plt.show()
