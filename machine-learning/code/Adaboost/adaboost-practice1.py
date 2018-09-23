# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer  #数据处理
from sklearn.model_selection import GridSearchCV 
from sklearn.ensemble import AdaBoostClassifier,GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
 

#读取数据
df = pd.read_csv('../datas/round1_train_countfeatures_test.csv')
print(df.head())
#划分训练集测试集
train = df[df['days'] <=23]
test = df[df['days'] == 24]
print(df.head())

#第一种，pandas+手动调参
"""
#缺失值处理--
#第一种，基于pandas的处理
df = df.replace(-1,np.nan)
df = df.fillna(df.mode().loc[0])
#第二种，基于sklearn，Imputer
#Im = Imputer(strategy='most_frequent')
#df = Im.fit_transform(df)

#提取x,y 
x_train = train.drop(['is_trade','instance_id'],1)
y_train = train['is_trade']
x_test = test.drop(['is_trade','instance_id'],1)
y_test = test['is_trade']
print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)


#建模
tree = DecisionTreeClassifier(max_depth=2)
ada = AdaBoostClassifier(tree,n_estimators=10,learning_rate=0.1)
ada.fit(x_train,y_train)
ada.score(x_test,y_test)
print(ada.score(x_test,y_test)) 
"""

#第二种 pipeline+自动调参写法
df = df.replace(-1,np.nan)
df = df.fillna(df.mode().loc[0])
x_train = train.drop(['is_trade','instance_id'],1)
y_train = train['is_trade']
x_test = test.drop(['is_trade','instance_id'],1)
y_test = test['is_trade']
tree = DecisionTreeClassifier()
pipe = Pipeline([
            ('Im',Imputer(strategy='most_frequent')),
            ('ada',AdaBoostClassifier(tree))
        ])
pipe1 = Pipeline([
            ('Im',Imputer()),
            ('GBDT',GradientBoostingClassifier())
        ])

#base_estimator分类器
params = {
        'Im__strategy':['most_frequent'],
        'ada__n_estimators':[50,100],
        'ada__learning_rate':[0.1,0.8],
        'ada__base_estimator__max_depth':[2]  
        }
params1 = {
        'Im__strategy':['most_frequent','median'],
        'GBDT__n_estimators':[50,100],
        'GBDT__learning_rate':[0.1,0.8],
        'GBDT__max_depth':[2]  
        }

#网格搜索调参
#model = GridSearchCV(pipe,params,scoring='neg_log_loss',cv=3) #scoring评价指标,model是一个pipeline
model = GridSearchCV(pipe1,params1,scoring='neg_log_loss',cv=3) #scoring评价指标,model是一个pipeline

model.fit(x_train,y_train)
print(model.best_params_)  #输出最优的参数字典
print(model.best_score_)  #输出最优的得分
model_best = model.best_estimator_  #获得最优的模型（pipeline)
#model_best = model_best.get_params()['ada'] #从pipeline里获得ada的参数属性
model_best = model_best.get_params()['GBDT'] #从pipeline里获得ada的参数属性
#print(model_best.estimator_weights)     #获得模型的分类器的权重属性
#params_new = {'ada__base_estimator__max_depth': 2, 'ada__learning_rate': 0.1, 'ada__n_estimators': 50}
