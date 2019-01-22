#encoding=utf8
from __future__ import print_function
from time import time
import logging
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC
print(__doc__)

#Display progress logs on stdout
logging.basicConfig(level=logging.INFO, format='%(asctime)s%(message)s')

# ################################################################
#Download the data, if not already on disk and load it as numpy arrays
lfw_people = fetch_lfw_people(min_faces_per_person=100, resize=0.4)

#introspect the images arrays to find the shapes (for plotting)

n_samples, h, w = lfw_people.images.shape

X = lfw_people.data
n_features = X.shape[1]

#the label to predict is the id of the person
y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]

print("Total dataset size:")
print("n_samples: %d" %n_samples)
print("n_features: %d" %n_features)
print("n_classes: %d" %n_classes)

# ################################################################
#Split into a training set and a test set using a stratified k fold

#split into a training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# ################################################################
# Compute a PCA on the face dataset 
n_components = 150 #设置PCA的主成分个数，类似svd的奇异值
print("Extracting the top %d eigenfaces from %d faces" %(n_components, X_train.shape[0]))
t0 = time()
pca = PCA(n_components=n_components, svd_solver='randomized',whiten=True)
pca = pca.fit(X_train)
print("done in %0.3fs" %(time() - t0))
eigenfaces = pca.components_.reshape((n_components, h, w))
print("Projecting the input data on the eigenfaces orthonormal basis")
t0 = time()
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

print("done in %0.3fs" %(time() - t0))

# ################################################################
#train a svm classification model
print("Fitting the classifier to the training set")
t0 = time()
param_grid = {'C':[1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma':[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],}
clf = GridSearchCV(SVC(class_weight='balanced'),param_grid,cv=5)
clf = clf.fit(X_train_pca, y_train)
print("done in %0.3fs" %(time() -t0))
print("Best estimator found by grid search:")
print(clf.best_estimator_)
# ################################################################
print("Predicting people's name on the test set")
t0 =time()
y_pred = clf.predict(X_test_pca)
print(classification_report(y_test, y_pred, target_names=target_names))
print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))
# ################################################################

def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8*n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i+1)
        plt.imshow(images[i].reshape((h,w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())

#plot the result of the prediction on a portion of the test set
def title(y_pred, y_test, target_names,i):
    pred_name = target_names[y_pred[i]].rsplit(' ',1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ',1)[-1]
    return 'predicted：%s\ntrue:    %s' %(pred_name, true_name)

prediction_titles = [title(y_pred, y_test, target_names, i) for i in range(y_pred.shape[0])]
plot_gallery(X_test, prediction_titles, h, w)

#plot the gallery of the most significative eigenfaces
eigenfaces_titles = ['eigenface %d' % i for i in range(eigenfaces.shape[0])]
plot_gallery(eigenfaces, eigenfaces_titles, h, w)
plt.show()

# #######################################
"""
step1:下载数据集
step2:取数据集的信息： 因为是图片数据，shape是3个维度，size,high,weight
取数据的X，Y，X的维度为size,特征数=high*weight
step3:划分训练集和测试集
step4:在训练集上训练PCA模型
step5:在训练和测试集上用训练好的pca模型进行转换
step6:通过网格搜索在训练集上训练出最优的svm模型分类器clf
step7:用clf对测试集进行预测，得到预测结果
step8:打印分类报告:在测试集上的精确率，召回率，F1-score
step9:打印混淆矩阵：在测试集上的混淆矩阵
step10:打印预测的照片,数据要转化为[h,w]的像素形式
step11:打印主成分的照片

"""
# #######################################
