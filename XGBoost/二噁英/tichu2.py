#-*- coding :utf-8-*-
import pandas as pd


data = pd.DataFrame(pd.read_csv('ereying2.csv'))
data.dropna(inplace=True)
data.to_csv('ereying4.csv')