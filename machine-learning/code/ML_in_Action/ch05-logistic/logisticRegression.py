#encoding=utf8
import numpy as np
 
 
def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open('testSet.txt')
    for line in  fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat
def sigmoid(inX):
    return 1.0/ (1+np.exp(-inX))
def gradAscent(dataMatIn, classLabels):
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()
    m,n = np.shape(dataMatrix)
    weights = np.ones((n, 1))
    alpha = 0.001
    maxCycles = 500
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)
        error = (labelMat - h)
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights

def plotBestFit(weights):
    """
    读取数据x,y
    将x转化为array,获取数据个数
    遍历数据用list分别存储分类的x，y
    创建figure
    在figure里面添加子图
    将分类的结果x,y分别用红和绿表示
    """
    import matplotlib.pyplot as plt
    dataMat, labelMat = loadDataSet()
    dataArr = np.array(dataMat)
    n = np.shape(dataArr)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
     
    x = np.arange(-3.0, 3.0, 0.1) #x为numpy.arange格式，并且以0.1为步长从-3.0到3.0切分。
    #拟合曲线为0 = w0*x0+w1*x1+w2*x2, 故x2 = (-w0*x0-w1*x1)/w2, x0为1,x1为x, x2为y,故有
    y = (-weights[0] - weights[1]*x)/weights[2]
    ax.plot(x, y)
    plt.xlabel('X1') ;plt.ylabel('X2');
    plt.show()

def stocGradAscent0(dataMatrix, classLabels):
    dataMatrix = np.array(dataMatrix)
    m,n =np.shape(dataMatrix)
    alpha = 0.01
    weights = np.ones(n)
    for i in range(m):
        h =sigmoid(sum(dataMatrix[i]*weights)) #list * list,w0x0+w1x1+w2x2
        error=classLabels[i] -h
        weights = weights + alpha *error *dataMatrix[i]
    return weights

def stocGradAscent1(dataMatrix, classLabels, numIter = 200):
    """
    1.动态减小学习率
    2.随机选取样本，并将样本删除
    """
    m, n = np.shape(dataMatrix)
    weights = np.ones(n)
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.0001
            randIndex = int(np.random.uniform(0,len(dataIndex)))
            h  = sigmoid(sum(dataMatrix[randIndex] * weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights
def classifyVector(inX, weights):
    """
    预测：
    结果大于0.5 返回1
    否则返回0
    """
    prob = sigmoid(sum(inX * weights))
    if prob > 0.5: 
        return 1.0
    else:
        return 0.0
    
def colicTest():
    """
    1、打开训练和测试文本
    2、读取训练的x,y
    3、用训练集采用随机梯度下降求w
    4、读取测试的x,y
    5、用测试集测试准确率
    """
    frTrain = open("horseColicTraining.txt",'r'); frTest=open("horseColicTest.txt")
    trainSet =[]; trainLabels=[]
    for line in frTrain.readlines():
        currline = line.strip().split("\t")
        lineArr = []
        for i in range(21):
            lineArr.append(float(currline[i]))
        trainSet.append(lineArr)
        trainLabels.append(float(currline[-1]))
    trainWeights = stocGradAscent1(np.array(trainSet), trainLabels, 1000)
    errorCount = 0.0; numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currline = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currline[i]))
        if int(classifyVector(np.array(lineArr), trainWeights)) != int(currline[-1]):
            errorCount += 1
    errorRate = (float(errorCount / numTestVec))
    print("the error rate of this test is :%f" % errorRate)
    return errorRate
def multiTest():
    numTests = 10; errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print("after %d iterations the average error rate is :%f" %(numTests, errorSum/float(numTests)))

#dataArr,labelMat = loadDataSet()
#weights = stocGradAscent0(np.array(dataArr), labelMat)   
#plotBestFit(weights)  #将numpy矩阵转换为数组
#weights = stocGradAscent1(np.array(dataArr), labelMat)    
#plotBestFit(weights)  #将numpy矩阵转换为数组
multiTest()