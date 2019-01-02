# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 21:31:53 2019

@author: hcc
"""
 
#encoding=utf8
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import warnings
import sklearn
from sklearn.linear_model import LogisticRegressionCV,LinearRegression
from sklearn.linear_model.coordinate_descent import ConvergenceWarning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
# 设置字符集，防止中文乱码
mpl.rcParams['font.sans-serif']=[u'simHei']
mpl.rcParams['axes.unicode_minus']=False
# 拦截异常
warnings.filterwarnings(action = 'ignore', category=ConvergenceWarning)
# 数据读取并处理异常数据
path = "datas/breast-cancer-wisconsin.data"
names = ['id','Clump Thickness','Uniformity of Cell Size','Uniformity of Cell Shape',
         'Marginal Adhesion','Single Epithelial Cell Size','Bare Nuclei',
         'Bland Chromatin','Normal Nucleoli','Mitoses','Class']

df = pd.read_csv(path, header=None,names=names)

datas = df.replace('?', np.nan).dropna(how = 'any') # 只要有列为空，就进行删除操作
print(datas.head(1)) # 显示一下
X = datas[names[1:10]] #取特征
Y = datas[names[10]] #取标签
X_train, X_test,Y_train, Y_test = train_test_split(X,Y,test_size=0.1,random_state =0)
ss = StandardScaler()#标准化
X_train = ss.fit_transform(X_train) #训练模型及归一化数据
lr = LogisticRegressionCV(multi_class='ovr', fit_intercept=True, Cs=np.logspace(-2, 2, 20), cv=2, penalty='l2', solver='lbfgs', tol=0.01)
re = lr.fit(X_train,Y_train)
r = re.score(X_train, Y_train)#模型效果 
print("准确率：",r)
print("稀疏化特征比率：%.2f%%" % (np.mean(lr.coef_.ravel()==0)*100))
print("参数：",re.coef_)
print("截距：",re.intercept_)
print(re.predict_log_proba(X_test))

joblib.dump(re,"datas/train_model.m") #保存模型
model = joblib.load("datas/train_model.m") #读取模型

#数据预测
X_test = ss.transform(X_test)
Y_predict = model.predict(X_test)
print(Y_predict)