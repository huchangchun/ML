# -*- coding: utf-8 -*-
"""
Created on Sun May 27 14:51:35 2018

@author: hcc
"""
import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import GradientBoostingClassifier
 
df = pd.read_csv('./datas/round1_train_countfeatures.csv')
x = df.drop(['is_trade'],1)
y = df['is_trade']
#ipp_ra_pp  是预测商品属性数目和广告商品属性数目的比值
print(x.isnull().sum())
 
x.fillna(x.mean(),inplace=True)
 
gbdt = GradientBoostingClassifier()
gbdt.fit(x,y)
#print(gbdt.feature_importence_)

sfm = SelectFromModel(gbdt,threshold=0.001)
sfm.fit(x,y)
x_gbdt = sfm.transform(x)
print(x_gbdt)
print(x.shape)
print(x_gbdt.shape)

"""
[[  1.08641075e+17   3.41272038e+18   1.97559044e+18 ...,   9.00000000e+00
    1.00000000e+01   1.80000000e+01]
 [  5.75471355e+18   3.41272038e+18   1.97559044e+18 ...,   9.00000000e+00
    1.20000000e+01   1.80000000e+01]
 [  8.42679481e+17   3.41272038e+18   1.97559044e+18 ...,   9.00000000e+00
    3.00000000e+00   1.80000000e+01]
 ..., 
 [  5.69377066e+18   5.04083465e+18   2.60381541e+18 ...,   8.00000000e+01
    2.00000000e+01   2.40000000e+01]
 [  4.62325319e+18   5.04083465e+18   2.60381541e+18 ...,   8.00000000e+01
    1.80000000e+01   2.40000000e+01]
 [  8.01312475e+18   3.76243748e+18   6.44776654e+18 ...,   2.10000000e+01
    1.90000000e+01   2.40000000e+01]]
(478138, 29)
(478138, 29)

"""