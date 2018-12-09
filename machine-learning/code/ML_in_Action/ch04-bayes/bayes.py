#encoding=utf-8
import numpy as np
def loadDataSet():
    postingList = [['my','dog','has','flea','problems','help','please'],
                   ['maybe','not','take','him','to','dog','park','stupid'],
                   ['my','dalmation','is','so','cute','I','love','him'],
                   ['stop','posting','stupid','worthless','garbage'],
                   ['mr','licks','ate','my','steak','how','to','stop','him'],
                   ['quit','buying','worthless','dog','food','stupid']]
    classVec = [0,1,0,1,0,1] #1代表侮辱性文字，0代表正常言论
    return postingList, classVec
def createVocabList(dataSet):
    vocabList = set([])
    for document in dataSet:
        vocabList = vocabList | set(document) #取并集
    return list(vocabList)

def setOfWords2Vec(vocabList, inputSet):
    """
    对词进行One-Hot编码,每个句子的词向量长度均为，词汇表的长度N，
    新建一个长度为N值全0的数组，查词汇表得到句子中每个词在词汇表的索引，数组对应索引置1
    """
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print('the word :%s is not in my vocabulary!' % word)
    return  returnVec
def trainNB(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs) #得到正例占所有样本的比率
    p0Num = np.ones(numWords)
    p1Num = np.ones(numWords)      #change to ones() 
    p0Denom = 2.0
    p1Denom = 2.0                        #change to 2.0
    for i in range(numTrainDocs):#遍历标签，分别对1/0两个类别统计，当类别为正例时，每个词的个数以及所有词的总数
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i]) 
    #对每个类别计算出每个词占所有词总数的比率并取对数
    p1Vect = np.log(p1Num/p1Denom)          #change to log()
    p0Vect = np.log(p0Num/p0Denom)          #change to log()
    return p0Vect, p1Vect, pAbusive
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify*p1Vec) + np.log(pClass1)
    p0 = sum(vec2Classify*p0Vec) + np.log(1 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0
def testingNB():
    dataSet , listClasses= loadDataSet()
    vocabList = createVocabList(dataSet)
    wordVec = setOfWords2Vec(vocabList, dataSet[0])
    trainMat = []
    for postinDoc in dataSet:
        trainMat.append(setOfWords2Vec(vocabList,postinDoc))
    p0V,p1V,pAb = trainNB(np.array(trainMat), np.array(listClasses))
    testEntry = ['love','my','dalmation']
    thisDoc = np.array(setOfWords2Vec(vocabList,testEntry))
    print(testEntry,'classified as:',classifyNB(thisDoc,p0V,p1V,pAb))
    testEntry = ['stupid','garbage']
    thisDoc = np.array(setOfWords2Vec(vocabList,testEntry))
    print(testEntry,'classified as:',classifyNB(thisDoc,p0V,p1V,pAb))
    
def bagOfWord2VecMN(vocabList, inputSet):
    """
    对词进行One-Hot编码,每个句子的词向量长度均为，词汇表的长度N，
    新建一个长度为N值全0的数组，查词汇表得到句子中每个词在词汇表的索引，数组对应索引置1
    """
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return  returnVec

def textParse(bigString):
    import re
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok)> 2]

def spamTest():
    docList = [];classList = []; fullText = []
    for i in range(1,26):
        wordList = textParse(open('email/spam/%d.txt' % i,encoding='utf-8').read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
  
        wordList = textParse(open('email/ham/%d.txt' % i,encoding='utf-8').read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    trainingSet =list(np.arange(50));testSet = []
    for i in range(10):
        randIndex = int(np.random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del (trainingSet[randIndex])
    trainMat = []; trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(bagOfWord2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = trainNB(np.array(trainMat), np.array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = bagOfWord2VecMN(vocabList, docList[docIndex])
        if classifyNB(np.array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1
            print("classification error:", classList[docIndex])
            print(docList[docIndex])
    print(testSet)
    print("the error rate is :", float(errorCount)/ len(testSet))
    return vocabList, fullText
    
def calcMostFreq(vocabList, fullText):
    pass 
            
if __name__=='__main__':
    testingNB()
    spamTest()
    """
    对训练样本，假设有m个A类，n个B类，词的总数分别为kA,KB
    首先求出对应于A类下每个词占A类总数的概率并取对数
    pA_i = log(w_i/KA),VectA为pA_i组成的矩阵
    pB_i = log(w_i/KB),VectB为pB_i组成的矩阵
    pCA = log(m/m+n)
    PCB = log(n/m+n)
    当来一句话时，通过训练样本词汇表长度创建一个0矩阵，矩阵对应训练样本的词通过查字典方式，同一个词出现多次时对应位置累加1
    得到矩阵Mat_P
    类别预测是分别计算A/B类的概率,p(c|w) = p(c)sum(p(wi|c))
    Pre_A =sum(Mat_P*VectA) + pCA//将VectA和Mat_p对应位置相乘
    Pre_B =sum(Mat_p*VectB) + pCB
    比较Pre_A和Pre_B的大小，取概率大的作为测试样本的分类
    所以贝叶斯模型时生成模式，通过联和概率求的样本的概率。
    """