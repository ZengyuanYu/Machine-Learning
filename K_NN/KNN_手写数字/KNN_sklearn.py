#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by XiaoYu on 18-3-8
# -*- coding: UTF-8 -*-
import numpy as np
import operator
import time
from os import listdir
from sklearn.neighbors import KNeighborsClassifier as KNN

"""
函数说明:将32x32的二进制图像转换为1x1024向量。

Parameters:
    filename - 文件名
Returns:
    returnVect - 返回的二进制图像的1x1024向量
"""
def img2vector(filename):
    #创建1x1024零向量
    returnVect = np.zeros((1, 1024))
    #打开文件
    fr = open(filename)
    #按行读取
    for i in range(32):
        #读取一行数据
        lineStr = fr.readline()
        #将每一行的前32个元素依次添加到returnVect中
        for j in range(32):
            returnVect[0, 32*i+j] = int(lineStr[j])
    #返回转换后的1x1024向量
    return returnVect

"""
函数说明:手写数字分类测试

Parameters:
    无
Returns:
    无
"""
def handwritingClassTest():
    #测试集的Labels
    hwLabels = []
    #返回traingDigits目录下的文件名
    trainingFileList = listdir('trainingDigits')
    #返回文件夹下文件的个数
    m = len(trainingFileList)
    #初始化训练的Mat矩阵，测试集
    trainingMat = np.zeros((m, 1024))
    #从文件名中解析处训练集的区别
    for i in range(m):
        #获得文件的名字
        fileNameStr = trainingFileList[i]
        #获得分类的数字
        classNumber = int(fileNameStr.split('_')[0])
        #将获得的类别添加到hwLabels中
        hwLabels.append(classNumber)
        #将每个文件的1x1024数据存储到trainingMat的矩阵中
        trainingMat[i, :] = img2vector('trainingDigits/%s' %(fileNameStr))
    #构建KNN分类器
    neigh = KNN(n_neighbors=3, algorithm='auto')
    #拟合模型，trainingMat为训练矩阵，hwLabels为对应标签
    neigh.fit(trainingMat, hwLabels)
    #返回testDigits目录下的文件列表
    testFileList = listdir('testDigits')
    #错误检测计数
    errorCount = 0.0
    #测试数据的数量
    mTest = len(testFileList)
    #从文件解析出测试集类别并进行分类训练
    for i in range(mTest):
        #获得文件的名字
        fileNameStr = testFileList[i]
        #获得分类的数字
        classNumber = int(fileNameStr.split('_')[0])
        #获得测试集的1x1024向量用于训练
        vectorUnderTest = img2vector('testDigits/%s' % (fileNameStr))
        #获得预测结果
        classifierResult = neigh.predict(vectorUnderTest)
        print("分类返回结果为%d\t真实结果为%d" % (classifierResult, classNumber))
        if (classifierResult != classNumber):
            errorCount += 1.0
    print("共错了%d个数据\n准确率为%f%%" %(errorCount, 100-errorCount/mTest*100))

if __name__ == '__main__':
    start = time.time()
    handwritingClassTest()
    print("总共运行了：%s s" % (time.time()-start))