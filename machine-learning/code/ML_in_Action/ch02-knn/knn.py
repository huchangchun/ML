#encoding=utf-8
import  numpy as np
import operator
import os
def createDataSet():
    group = np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels
def classfiy0(inX,dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX,(dataSetSize,1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distance = sqDistances ** 0.5
    sortedDistIndices = distance.argsort()#从小打到排序后，返回数据的小标array([2, 3, 1, 0], dtype=int64)
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndices[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.items(),key = operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0] #[('B', 2), ('A', 1)]返回'B'
group ,labels = createDataSet()
#classfiy0([0,0],group,labels,3)

def file2Matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numbefOfLines = len(arrayOLines)
    returnMat = np.zeros((numbefOfLines,3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[:3]
        classLabelVector.append(int(listFromLine[-1]))# 需要明确告诉解释器，转化为整形否则就是字符
        index += 1
    return returnMat, classLabelVector

#归一化
def autoNorm(dataSet):
    """
    newValue = (oldValue - min)/(max-min)
    """
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals,(m,1))
    normDataSet = normDataSet / np.tile(ranges,(m,1))
    return normDataSet,ranges,minVals
def datingClassTest(filename):
    hoRatio = 0.10
    datingDataMat,datingLabels = file2Matrix(filename)
    normMat,ranges ,minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVec = int(m*hoRatio)
    errorCount = 0.0
    """
    取0-100个样本，然后通过分类器处理得到预测标签，再与实际标签判断，并计算错误率
    """
    for i in range(numTestVec):
        
        classifierResult = classfiy0(normMat[i,:],normMat[numTestVec:m,:],datingLabels[numTestVec:m],3)
        print("the classifier name came back with :%d,the real answer is :%d" %(classifierResult,datingLabels[i]))
        if(classifierResult != datingLabels[i]):
            errorCount += 1.0
    print("the total error rate is :%f" %(errorCount/float(numTestVec)))
    
    import matplotlib
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(datingDataMat[:,1],datingDataMat[:,2],15*np.array(datingLabels),15*np.array(datingLabels))
    plt.show()
def img2Vector(filename):
    returnVect = np.zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        linestr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(linestr[j])
    return returnVect

def handWritingClassTest():
    """
    读取训练目录下的文件list，得到m个文件，文件名为表示为分类数字的第n个例子
    创建一个（m,1024)的矩阵，将文件中的数据读入到矩阵中，每一行存储一个图像，然后记录标签数据
    读取测试目录下的文件，每次读取一个文件后，将数据进行预测
    """
    hwLabels = []
    trainingFileList = os.listdir(path = 'digits/trainingDigits')
    m = len(trainingFileList)
    trainingMat = np.zeros((m,1024))
    for i in range(m):
        fileStr = trainingFileList[i]
        hwLabels.append(int(fileStr.split('_')[0]))
        trainingMat[i,:] = img2Vector('digits/trainingDigits/%s' %fileStr)
    testFileList = os.listdir(path='digits/testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        classNum = int(testFileList[i].split("_")[0])
        TestinX = img2Vector('digits/testDigits/%s' % testFileList[i])
        classifierResult = classfiy0(TestinX,trainingMat, hwLabels, 3)
        #print("the classifier came back with :%d,the real answer is :%d" %(classifierResult,classNum))
        if(classifierResult != classNum):
            errorCount += 1.0
    print("the total number of errors is %d " % errorCount)
    print("the total error rate is :%f" % (errorCount/float(mTest)))
        
if __name__ =='__main__':
    
    #datingClassTest('datingTestSet2.txt')
    #testVector = img2Vector('digits/testDigits/0_13.txt')
    handWritingClassTest()
  