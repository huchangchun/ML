from sklearn.datasets import make_multilabel_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
#def make_multilabel_classification(n_samples=100, n_features=20, n_classes=5,
                                   #n_labels=2, length=50, allow_unlabeled=True,
                                   #sparse=False, return_indicator='dense',
                                   #return_distributions=False,
                                   #random_state=None):
    #"""Generate a random multilabel classification problem.

    #For each sample, the generative process is:
        #- pick the number of labels: n ~ Poisson(n_labels)
        #- n times, choose a class c: c ~ Multinomial(theta)
        #- pick the document length: k ~ Poisson(length)
        #- k times, choose a word: w ~ Multinomial(theta_c)
#X, Y = make_multilabel_classification(n_samples=30000, sparse=True, n_labels=2,n_classes=30, return_indicator='sparse', allow_unlabeled=False)
 
#X_train,X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=33)

from sklearn.externals import joblib
X_train,X_test, y_train, y_test = joblib.load('./multi_label_dataset_0527')
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.naive_bayes import GaussianNB
from skmultilearn.problem_transform import ClassifierChain
from skmultilearn.problem_transform import LabelPowerset
from skmultilearn.adapt import MLkNN
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC


#model = {
         #'BinaryRelevance': BinaryRelevance(GaussianNB()),
         #'ClassifierChain': ClassifierChain(GaussianNB()),
         #'LabelPowerset': LabelPowerset(GaussianNB()),
         #'MLkNN': MLkNN(k=20),
         #"SVC": OneVsRestClassifier(LinearSVC()),
         #}
 
model = joblib.load('MLkNN')
def acc(pred, y):
    import numpy as np
    countAllEqual = 0
    countNotAllEqual = 0
    for i in range(len(pred)):
        if np.sum(np.equal(pred[i],y[i])) == len(pred[i]):
            countAllEqual += 1
        if np.sum(np.multiply(pred[i],y[i])) > 0:
            countNotAllEqual += 1
            #print("match: {}/{}".format(np.sum(np.multiply(pred[i],y[i])),(np.sum(y[i]))))
    
    return countAllEqual/len(pred), countNotAllEqual/len(pred)
#model.fit(X_train, y_train)
predictions = model.predict(X_test)
score = accuracy_score(y_test,predictions)
model_name = 'MLkNN'
print("Model: {:<12}   score: {:<12.2f} ".format(model_name,score))    
allequal, notallequal = acc(predictions.toarray(), y_test)
print("allequal: {:<12.2f}  notallequal: {:<12.2f}\n".format(allequal, notallequal))


#for model_name, model in model.items():
    #model.fit(X_train, y_train)
    #predictions = model.predict(X_test)
    #score = accuracy_score(y_test,predictions)
    #print("Model: {:<12}   score: {:<12.2f} ".format(model_name,score))    
    #allequal, notallequal = acc(predictions.toarray(), y_test.toarray())
    #print("allequal: {:<12.2f}  notallequal: {:<12.2f}\n".format(allequal, notallequal))
    