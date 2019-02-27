#encoding=utf-8
def loadDataSet():
    return [[1,3,4],
            [2,3,5],
            [1,2,3,5],
            [2,5]]
def createC1(dataSet):
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])
    C1.sort()
    return list(map(frozenset,C1))
def scanD(D,Ck,minSupport):
    """
    获取每个子集出现在数据集中的次数用字典存放
    计算每个子集的支持度，返回支持度大于minSupport的子集
    frozenset可以作为字典的key,set不行
    """
    ssCnt = {}
    for tid in D:
        for can in Ck:
            if can.issubset(tid):
                if can not in ssCnt:ssCnt[can] = 1
                else:ssCnt[can]+= 1
    
    numItems = float(len(D))
    retList = []
    supportData = {}
    for key in ssCnt:
        support = ssCnt[key] / numItems
        if support >= minSupport:
            retList.insert(0,key) #前插法
        supportData[key] = support
    return retList, supportData
def aprioriGen(Lk, k): #creates Ck
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i+1, lenLk): 
            L1 = list(Lk[i])[:k-2]; L2 = list(Lk[j])[:k-2]
            L1.sort(); L2.sort()
            if L1==L2: #if first k-2 elements are equal
                retList.append(Lk[i] | Lk[j]) #set union
    return retList
 
def apriori(dataSet, minSupport=0.5):
    """
    关联分析的目标分为两个：
    发现频繁项集和发现关联规则
    首先需要找到频繁项集，然后才能获得关联规则
    Apriori算法是发现频繁项集的一种方法：Apriori算法的两个输入参数分别是最小支持度和数据集，
    该算法首先会生成所有单个物品的项集列表，接着扫描哪些项集满足最小支持度要求，
    那些不满足最小支持度的集合会被去掉，然后对剩下的集合进行组合以生成包含两个元素的项集，接下来，再重新扫描
    数据集，去掉不满足最小支持度的项集，该过程重复进行直到所有项集都被去掉。
    """
    C1 = createC1(dataSet)
    D = list(map(set, dataSet))
    L1, supportData = scanD(D, C1, minSupport)
    L = [L1]
    k = 2
    while (len(L[k-2]) > 0):
        Ck = aprioriGen(L[k-2],k)
        Lk,supK = scanD(D, Ck, minSupport)
        supportData.update(supK)
        L.append(Lk)
        k += 1
    return L, supportData
def generateRules(L, supportData, minConf=0.7):  #supportData is a dict coming from scanD
    bigRuleList = []
    for i in range(1, len(L)):#only get the sets with two or more items
        for freqSet in L[i]:
            H1 = [frozenset([item]) for item in freqSet]
            if (i > 1):
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList         

def calcConf(freqSet, H, supportData, brl, minConf=0.7):
    prunedH = [] #create new list to return
    for conseq in H:
        conf = supportData[freqSet]/supportData[freqSet-conseq] #calc confidence
        if conf >= minConf: 
            print(freqSet-conseq,'-->',conseq,'conf:',conf)
            brl.append((freqSet-conseq, conseq, conf))
            prunedH.append(conseq)
    return prunedH

def rulesFromConseq(freqSet, H, supportData, brl, minConf=0.7):
    m = len(H[0])
    if (len(freqSet) > (m + 1)): #try further merging
        Hmp1 = aprioriGen(H, m+1)#create Hm+1 new candidates
        Hmp1 = calcConf(freqSet, Hmp1, supportData, brl, minConf)
        if (len(Hmp1) > 1):    #need at least two sets to merge
            rulesFromConseq(freqSet, Hmp1, supportData, brl, minConf) 
def pntRules(ruleList, itemMeaning):
    for ruleTup in ruleList:
        for item in ruleTup[0]:
            print(itemMeaning[item])
        print("           -------->")
        for item in ruleTup[1]:
            print(itemMeaning[item])
        print("confidence: %f" % ruleTup[2])
        print()       #print a blank line
                    
if __name__ =='__main__':
    dataSet = loadDataSet()
    C1 = createC1(dataSet)
    D = list(map(set,dataSet))
    scanD(D, C1, 0.5)
    print(dataSet)
    print(D)
 
    L, supportData =apriori(dataSet)
    
    rules= generateRules(L, supportData)
    print(rules) 
    
    rules= generateRules(L, supportData,0.5)
    print("=====================")
    for r in rules:
        print(r)