#! /usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Xiaoyu

# Import SVC
from sklearn.svm import SVR
import pandas as pd
# Create a support vector classifier
clf = SVR()

X_train = pd.read_csv('X_train.csv')
X_test = pd.read_csv('X_test.csv')
y_train = pd.read_csv('y_train.csv')['0']
y_test = pd.read_csv('y_test.csv')['0']


print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
# # Fit the classifier using the training data
clf.fit(X_train, y_train)
#
# # Predict the labels of the test set
y_pred = clf.predict(X_test)

# Count the number of correct predictions
n_correct = 0
print(y_test)
print(y_pred)

from sklearn.metrics import r2_score

print(r2_score(y_test, y_pred))
