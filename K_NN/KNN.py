#-*- coding: UTF-8 -*-
import numpy as np
import operator
"""
函数说明:打开并解析文件，对数据进行分类：1代表不喜欢,2代表魅力一般,3代表极具魅力

Parameters:
    filename - 文件名
Returns:
    returnMat - 特征矩阵
    classLabelVector - 分类Label向量
"""
def file2matrix(filename):
    #打开文件
    fr = open(filename)
    #读取文件所有内容
    array0Lines = fr.readlines()
    #得到文件行数
    numberOfLines = len(array0Lines)
    #定义返回矩阵，解析完成的数据：numberOfLines行，3列
    returnMat = np.zeros((numberOfLines, 3))
    #返回分类标签向量
    classLabelVector = []
    #行的索引值
    index = 0
    for line in array0Lines:
        #s.strip(rm)，当rm空时,默认删除头尾空白符(包括'\n','\r','\t',' ')
        line = line.strip()
        #使用s.split(str="",num=string,cout(str))将字符串根据'\t'分隔符进行切片
        listFromLine = line.split('\t')
        #将数据前三列提取出来,存放到returnMat的NumPy矩阵中,也就是特征矩阵
        returnMat[index, :] = listFromLine[0:3]
       #根据文本中标记的喜欢的程度进行分类,1代表不喜欢,2代表魅力一般,3代表极具魅力
        if listFromLine[-1] == 'didntLike':
            classLabelVector.append(1)
        elif listFromLine[-1] == 'smallDoses':
            classLabelVector.append(2)
        elif listFromLine[-1] == 'largeDoses':
            classLabelVector.append(3)
        index += 1
    return returnMat, classLabelVector 

"""
函数说明:对数据进行归一化

Parameters:
    dataSet - 特征矩阵
Returns:
    normDataSet - 归一化后的特征矩阵
    ranges - 数据范围
    minVals - 数据最小值
"""
def autoNorm(dataSet):
    #获得数据的最值
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    #最大值和最小值的差值
    ranges = maxVals - minVals
    #shape(dataSet)返回dataSet矩阵的行列数,此时normDataSet是[100x3]的0矩阵
    normDataSet = np.zeros(np.shape(dataSet))
    #返回dataSet的行数
    m = dataSet.shape[0]
    #原始值减去最小值，np.tile（）是矩阵复制的一种方法，作用是将minVals复制成为m行1列的矩阵
    normDataSet = dataSet - np.tile(minVals, (m, 1))
    #除以最大值和最小值之差，得到归一数据
    normDataSet = normDataSet / np.tile(ranges, (m,1))
    #返回归一化数据结果，数据范围和最小值
    return normDataSet, ranges, minVals

"""
函数说明:kNN算法,分类器

Parameters:
    inX - 用于分类的数据(测试集)
    dataSet - 用于训练的数据(训练集)
    labes - 分类标签
    k - kNN算法参数,选择距离最小的k个点
Returns:
    sortedClassCount[0][0] - 分类结果
"""
def classify0(inX, dataSet, labels, k):
    #numpy函数shape[0]返回dataSet的行数
    dataSetSize = dataSet.shape[0]
    #在列向量方向上重复inX共1次（横向）,行向量方向重复inX共dataSetSize次
    #inX  [1x3] np.tile(inX, (dataSetSize, 1))变为[dataSetSize x 3]
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    #二维特征相减后平方
    sqDiffMat = diffMat**2
    #sum()所有元素相加，sum(0)列相加，sum(1)行相加
    sqDistance = sqDiffMat.sum(axis=1)
    #开方 计算处距离
    distances = sqDistance**0.5
    #返回distances中元素从小到大排序后的索引值
    sortedDistIndices = distances.argsort()
    #定义一个记录类别次数的字典
    classCount = {}
    for i in range(k):
        #取前k个元素的类别
        voteIlabel = labels[sortedDistIndices[i]]
        #计算类别次数
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    #key=operator.itemgetter(1)根据字典的值进行排序
    #key=operator.itemgetter(0)根据字典的键进行排序
    #reverse降序排序字典
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    #返回次数最多的类别,即所要分类的类别
    return sortedClassCount[0][0]

"""
函数说明:通过输入一个人的三维特征,进行分类输出

Parameters:
    无
Returns:
    无
"""
def classifyPerson():
    #输出结果
    resultList = ['讨厌', '有些喜欢', '非常喜欢']
    #三维特征用户输入
    precentTats = float(input("玩游戏所耗时间百分比:"))
    ffMiles = float(input("每年获得的飞行常客的里程数:"))
    iceCream = float(input("每周消费的冰激凌公升数:"))
    #打开文件
    filename = "K_NNData.txt"
    #处理文件
    datingDataMat, datingLabels = file2matrix(filename)
    #训练集归一化
    normMat, ranges, minVals = autoNorm(datingDataMat)
    #测试数据生成numpy数组
    inArr = np.array([ffMiles, precentTats, iceCream])
    #测试集归一化
    normArr = (inArr - minVals) / ranges
    #返回分类结果
    classifierResult = classify0(normArr, normMat, datingLabels, 4)
    #打印结果
    print("你可能 %s 这个人" %(resultList[classifierResult-1]))
    
if __name__ == '__main__':
    classifyPerson()
