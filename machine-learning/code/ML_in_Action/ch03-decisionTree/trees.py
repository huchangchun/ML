#coding=utf-8
from math import log
import numpy as np
import operator

"""
熵用来度量数据的混乱程度，熵越大，数据越混乱，数据集的熵 = 每个分类数据的熵的总和
单个分类数据的熵 = -plog(2,p) ，其中p为该分类占整个数据集的概率
在划分数据集前后信息发生的变化成为信息增益，信息增益越大，数据越有序
信息增益 = 原始数据集的熵 - 数据划分后的熵，通过公式可以看出划分后熵越小，信息增益越大
"""

def createDataSet():
    dataSet = [[1,1,'Yes'],
               [1,1,'Yes'],
               [1,0,'No'],
               [0,1,'No'],
               [0,1,'No'],               
               ]
    labels = ['no surfacing','flippers']
    return dataSet,labels
def calcShannonEnt(dataSet):
    """
    创建一个标签的词典
    计算每个标签出现的次数，计算出现的概率，计算出香农熵，然后累加起来
    """
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec  in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts:
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for k,v in labelCounts.items():
        prob = float(v)/numEntries
        shannonEnt -=  prob * log(prob,2)
    return shannonEnt
def splitDataSet(dataSet, axis, value):
    """
    @param:待划分的数据集
    @param:划分数据集的特征
    @param:需要返回的特征的值
    """
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet
def chooseBestFeatureToSplit(dataSet):
    """
    选择信息增益最大的特征作为划分
    首先计算总的信息熵
    然后遍历每个特征，按该特征划分得到子数据集，求子集的信息熵，

    """
    numFeatures = len(dataSet[0]) - 1      #the last column is used for the labels
    baseEntropy = calcShannonEnt(dataSet)#.9709505944546686
    bestInfoGain = 0.0; bestFeature = -1
    for i in range(numFeatures):        #iterate over all the features
        featList = [example[i] for example in dataSet]#create a list of all the examples of this feature
        uniqueVals = set(featList)       #get a set of unique values
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)     
        infoGain = baseEntropy - newEntropy     #calculate the info gain; ie reduction in entropy
        print("the %d: infoGain:%f" %(i,infoGain))
        if (infoGain > bestInfoGain):       #compare this to the best gain so far
            bestInfoGain = infoGain         #if better than current best, set to best
            bestFeature = i

    return bestFeature                      #returns an intege
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classList:
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]
def createTree(dataSet,labels):
    """
    1.获取分类标签列表
    2.类别完全相同则停止继续划分，返回类别
    3.遍历完所有特征时返回出现次数最多的类别
    4.选择信息增益最高的特征
    5.取特征对应的标签
    6.创建字典，以特征作为key
    7.删除最好的特征
    8.获取特征对应的所有属性值
    9.获取唯一属性值集合
    10.遍历所有属性值
       递归树赋值给字典
    """
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet) == 1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet,bestFeat,value),subLabels)
    return myTree
def classify(inputTree, featLabels, testVec):
    for k,v in enumerate(inputTree):
        firstStr = v
        break    
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel
def storeTree(inputTree, filename):
    import pickle
    fw = open(filename,'wb')
    pickle.dump(inputTree, fw)
    fw.close()
def grabTree(filename):
    import pickle
    with open(filename,'rb') as fr:
        return pickle.load(fr)
if __name__=='__main__':
    import treePlotter
    filename = './classStorage.txt'
    #myTree = treePlotter.retrieveTree(0)
    #leafs = treePlotter.getNumLeafs(myTree)
    #depth = treePlotter.getTreeDepth(myTree)
    #print("leaf:",leafs)
    #print("depth:",depth)
    #treePlotter.createPlot(myTree)
    dataSet,labels = createDataSet()
    shannonEnt = calcShannonEnt(dataSet)
    #print(splitDataSet(dataSet,0,0))#[[1, 'No'], [1, 'No'], [0, 'No']]
    #print(splitDataSet(dataSet,0,1))#[[1, 'Yes'], [1, 'Yes'], [0, 'No']]

    #print(chooseBestFeatureToSplit(dataSet))
    label = labels[:]
    myTree = createTree(dataSet,labels)
    print(myTree)
    storeTree(myTree,filename)
    inputTree = grabTree(filename)
    label = classify(inputTree, label, [1,1])
    print(label)
