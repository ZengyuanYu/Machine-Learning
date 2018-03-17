## 机器学习相关算法研究和复现
**2018-03-06**
### K-近邻算法
- 1.算法讲解
   - *K-近邻（K-NN）算法研究和实现.ipynb*算法讲解与Python实现
- 2.案例实现（1）
   - *Data格式.png*为数据存放格式
   - *K_NNData.txt*为数据存放
   - *KNN.py*KNN实现“海伦约会”对象的判定
- 3.案例实现（2）
   - *KNN_sklearn.py*使用机器学习库来对手写数字的分类
   - *testDigits/trainingDigits/*为手写数字数据集的测试集和训练集，其中每个txt文件为32x32个像素值
   

**2018-03-17**
### XGBoost算法

- 1.数据的预处理以及skleaen包进行XGBoost实现
   - **XGBoost使用.ipynb**介绍具体使用和文件处理

- 2.XGBoost预测台风实例
   - **xgboost数据预处理.ipynb**将*data*文件夹中的*.mat*文件转化成为*.csv*文件，其中，必须在同一文件夹下才能正常运行；
   
   - ~~数据文件为*X_noyear.csv*和*Y_noyear.csv*两个，数据类型为(66, 498)，(66,)；~~
   - **XGBoost预测台风.ipynb**为主要文件，里面包含对XGBmodel的构建以及具体训练过程；
   - 数据文件更改为*X_50.csv*,*X_16.csv*,*Y_50.csv*和*Y_16.csv*，分开来的训练集和测试机，改变原来的预测结构
   - **model**文件夹下存放.pkl文件

