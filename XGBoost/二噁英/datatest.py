#-*- coding :utf-8-*-
from functools import reduce

import pandas as pd

ereying='二噁英工控数据（1#炉）.xlsx'#二噁英数据

# data=pd.read_excel(ereying,sheetname=u'CO的浓度',index_col=u'时间')#读取数据，指定“尾部烟道进口烟气温度”的值为索引列

# print(data.describe())
'''
data1=pd.read_excel(ereying,sheetname=u'CO的浓度')
data1.to_csv('data1.csv',encoding='utf-8')

data2=pd.read_excel(ereying,sheetname=u'尾部烟道进口烟气温度')
data2.to_csv('data2.csv',encoding='utf-8')
'''


data1=pd.read_excel(ereying,sheetname=u'尾部烟道进口烟气温度')
data2=pd.read_excel(ereying,sheetname=u'炉尾部烟道出口烟气温度')
data3=pd.read_excel(ereying,sheetname=u'炉膛出口O2浓度1')
data4=pd.read_excel(ereying,sheetname=u'炉膛出口O2浓度2')
data5=pd.read_excel(ereying,sheetname=u'CO的浓度')
data6=pd.read_excel(ereying,sheetname=u'炉膛第一烟道出口烟气温度')
data7=pd.read_excel(ereying,sheetname=u'炉膛第一烟道入口烟气温度')
data8=pd.read_excel(ereying,sheetname=u'活性炭喷射量')
data9=pd.read_excel(ereying,sheetname=u'一次风机实际运行电流')
data10=pd.read_excel(ereying,sheetname=u'二次风机实际运行电流')
data11=pd.read_excel(ereying,sheetname=u'锅炉蒸汽量')

# all=data1.join([data2,data3,data4],on=u'时间',
#              how='outer')
# all.to_excel('all2.xlsx')
#锅炉的初始值前面是0，所以excel文件之中前半部分不出现
print("锅炉蒸汽量", data11)

dfs=[data1,data2,data3,data4,data5,data6,data7,data8,data9,data10,data11]
df_final=reduce(lambda left,right:pd.merge(left,right,on='时间',how='outer'), dfs)
names = [u'时间',u'尾部烟道进口烟气温度',u'炉尾部烟道出口烟气温度',u'炉膛出口O2浓度1',u'炉膛出口O2浓度2',u'CO的浓度',u'炉膛第一烟道出口烟气温度',
         u'炉膛第一烟道入口烟气温度',u'活性炭喷射量',u'一次风机实际运行电流',u'二次风机实际运行电流',u'锅炉蒸汽量']
df_final.columns = names
df_final.to_excel('all2.xlsx')
