#encoding=utf8
 
from sklearn.svm import SVC
import numpy as np
X = np.array([[-1, -1],[-2, -1],[1, 1],[2,1]])
y = np.array([1,1,2,2])
clf = SVC(gamma='auto')
clf.fit(X,y)
print(clf.predict([[-0.8,-1]]))
print(clf.class_weight_)
print(clf.support_)