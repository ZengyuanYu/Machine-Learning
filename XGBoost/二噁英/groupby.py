#-*- coding :utf-8-*-
import pandas as pd
ereying=pd.DataFrame(pd.read_excel('all2.xlsx'))
ereying=ereying.set_index(u'时间')

ereying2=ereying.resample('H').sum()
print(ereying2)
ereying2.to_csv('ereying2.csv',encoding='utf-8')

ereying=pd.DataFrame(pd.read_excel('all2_label.xlsx'))
ereying=ereying.set_index(u'时间')

ereying2=ereying.resample('H').sum()

ereying2.to_csv('ereying2_label.csv',encoding='utf-8')