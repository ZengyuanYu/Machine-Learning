#-*- coding :utf-8-*-
import pandas as pd
from sympy.abc import lamda

# for i in range(len(u'时间')):
#     filter(lamda len():len()>5)
#     TiChu=list(ereying2)
#

data = pd.DataFrame(pd.read_csv('ereying2.csv'))
# print(data.head().index)

# print(data.shape)
# data.drop(data.count(axis=1)<=5,inplace=True)
# ererying3.to_csv('ererying3.csv')
# print(data.count(axis=1)<=6)

# for indexs in data.head().index:
#     if data.count(axis=1)<=5:
#
#     # print(data.loc[indexs].values[0:11])
#      print(data.loc[indexs].values[0:11])

# data['count']=data.count(axis=1)
# # print(data.head())
# data.drop(data['count']==11,inplace=True)
# print(data.head(20))

# data.drop(data.count(axis=1)<=5,inplace=True)
data['count']=data.count(axis=1)<=6
# print(data.head())
data.drop(data['count']==True,inplace=True)
data.to_csv('ereying.csv')

print(data.head())

