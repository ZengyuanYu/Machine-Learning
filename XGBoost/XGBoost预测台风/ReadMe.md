## XGBoost预测台风
### 简单介绍
1. **xgboost数据预处理.ipynb**将*data*文件夹中的*.mat*文件转化成为*.csv*文件，其中，必须在同一文件夹下才能正常运行；
2. 数据文件为*X_noyear.csv*和*Y_noyear.csv*两个，数据类型为(66, 498)，(66,)；
3. **XGBoost预测台风.ipynb**为主要文件，里面包含对XGBmodel的构建以及具体训练过程；
4. **model**文件夹下存放.pkl文件

### 运行
> python XGBoost.py
