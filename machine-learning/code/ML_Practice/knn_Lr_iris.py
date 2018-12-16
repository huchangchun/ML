import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model.coordinate_descent import ConvergenceWarning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import label_binarize
from sklearn import metrics
import warnings
# 设置字符集，防止中文乱码
mpl.rcParams['font.sans-serif']=[u'simHei']
mpl.rcParams['axes.unicode_minus']=False
# 拦截异常
warnings.filterwarnings(action = 'ignore', category=ConvergenceWarning)

path = "../datas/iris.data"
names = ['sepal length', 'sepal width', 'petal length', 'petal width', 'cla']
df =pd.read_csv(path, header = None,names=names)
df['cla'].value_counts()
print(df.head(3))
def parseRecord(record):
    print('record:',record)
    result=[]
    r = zip(names,record)
    for name,v in r:
        if name == 'cla':
            if v == 'Iris-setosa':
                result.append(1)
            elif v == 'Iris-versicolor':
                result.append(2)
            elif v == 'Iris-virginica':
                result.append(3)
            else:
                result.append(np.nan)
        else:
            result.append(float(v))
    return result
datas = df.apply(lambda r: parseRecord(r), axis=1) #读取每一行axis=1按行，并重新赋值
print(datas)
datas=datas.dropna(how='any')
X=datas[names[0:-1]] #取0到倒二
Y=datas[names[-1]]
#划分训练集和测试集
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.4,random_state=0)
#标准化
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.fit_transform(X_test)
#特征选择
#降维处理
#模型构建
lr = LogisticRegressionCV(Cs=np.logspace(-4, 1,50), fit_intercept=True,
                          cv=3, dual=False, penalty='l2', scoring=None, solver='lbfgs', tol=1e-2,
                          max_iter=100, class_weight=None, n_jobs=1, verbose=0, refit=True,
                          intercept_scaling=1., multi_class='multinomial', random_state=None)
lr.fit(X_train,Y_train)

#将正确的数据转换为矩阵形式
y_test_hot = label_binarize(Y_test, classes=(1,2,3))
#得到预测的损失值
lr_y_score = lr.decision_function(X_test)
#计算ROC的值,ROC(i,j) 横纵分别对应假阳率，真阳率
lr_fpr,lr_tpr,lr_threasholds = metrics.roc_curve(y_test_hot.ravel(), lr_y_score.ravel())
#计算AUC的值,roc曲线下的面积
lr_auc = metrics.auc(lr_fpr,lr_tpr)
 
print("Logistic R值：",lr.score(X_train,Y_train))#准确率
print("Logistic算法auc值:",lr_auc)
lr_y_predict = lr.predict(X_test)
print(lr_y_predict)
print(lr.score(X_test,Y_test))

#KNN算法实现
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,Y_train)
#将正确的数据转换为矩阵形式
y_test_hot = label_binarize(Y_test,classes=(1,2,3))
#得到预测的损失值
knn_y_score = knn.predict_proba(X_test)
#计算roc的值,fpr：假阳率，tpr真阳率
knn_fpr,knn_tpr,knn_threasholds = metrics.roc_curve(y_test_hot.ravel(),knn_y_score.ravel())
#计算auc的值
knn_auc = metrics.auc(knn_fpr, knn_tpr)
print("KNN R值：",knn.score(X_train,Y_train))#准确率
print("KNN 算法auc值:",knn_auc)

#模型预测
knn_y_predict = knn.predict(X_test)
print(knn_y_score)
print(knn_fpr)

#画图

plt.figure(figsize=(8,6), facecolor='w')
plt.plot(lr_fpr,lr_tpr, c='r', lw=2 ,label=u'Logistic ,AUC=%.3f' % lr_auc)
plt.plot(knn_fpr, knn_tpr, c='g', lw=2, label=u'KNN, AUC=%.3f' %knn_auc)
plt.plot((0,1),(0,1),c='#a0a0a0',lw=2,ls='--')
plt.xlim(-0.01,1.02) #设置X轴的最大最小值
plt.ylim(-0.01,1.02) #设置y轴的最大最小值
plt.xticks(np.arange(0, 1.1,0.1))
plt.yticks(np.arange(0, 1.1,0.1))
plt.xlabel('False Positive Rate(FPR)',fontsize=16)
plt.ylabel('True Positive Rate(TPR)', fontsize=16)
plt.grid(b = True,ls = ':')
plt.legend(loc='Lower right', fancybox=True, framealpha=0.8, fontsize=12)
plt.title(u'莺尾花数据Logistic和KNN算法的ROC/AUC',fontsize=18)
#plt.show()

#预测结果画图
x_test_len = range(len(X_test))
plt.figure(figsize=(12,9), facecolor='w')
plt.ylim(0.5, 3.5)
plt.plot(x_test_len,Y_test,'ro',markersize=6, zorder=3,label=u'真实值')
plt.plot(x_test_len, lr_y_predict, 'g*', markersize = 10, zorder=2, label=u'Logis算法预测值,$R^2$=%.3f' % lr.score(X_test, Y_test))
plt.plot(x_test_len, knn_y_predict, 'b+', markersize = 16, zorder=1, label=u'KNN算法预测值,$R^2$=%.3f' % knn.score(X_test, Y_test))
plt.legend(loc = 'lower right')

plt.xlabel(u'数据编号', fontsize=18)
plt.ylabel(u'种类', fontsize=18)
plt.title(u'鸢尾花数据分类', fontsize=20)
plt.show()