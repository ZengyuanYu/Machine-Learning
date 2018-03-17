## XGboost分类器实现台风预测
### 文件介绍
- 1. *hgt.mat*,*sst.mat*,*vort.mat*为MATLAB格式的原始数据
- 2. *xgboost数据预处理.ipynb*将（1）中数据进行处理，输出为`.csv`文件
- 3. *hgt.csv*,*sst.csv*,*vort.csv*为转化之后的数据文件
- 4. *X\*.csv*和*Y\*.csv*是纯净的数据格式，可以直接使用sklearn数据包去读取
### XGBoost使用.ipynb
- 1. 根据Kaggle中对泰坦尼克号上人员的数据信息，通过三种变量来预测人员脱险概率；
- 2. 仿照（1）来进行预测，X为66*498格式的数据，其中498个影响因子决定了这一年的台风次数；
- 3. 使用sklearn数据读取包将数据分为`X_train`,`X_test`,`y_train`,`y_test`送入到分类器中 
