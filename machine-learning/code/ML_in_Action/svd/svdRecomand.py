# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 16:39:13 2018

@author: hcc
"""

#encoding=utf8
import numpy as np
import os,sys 
from numpy import linalg as la
root_path = os.path.abspath("../../")
if root_path not in sys.path:
    sys.path.append(root_path)
from Util.Timing import Timing
tim = Timing()

def loadExData2():
    """
    行表示用户
    列表示物品
    每一行中的第i个0表示该用户第i个物品没有评分
    """
    return [[2, 0, 0, 0, 2, 0, 0, 0, 4, 4, 0, 0, 0, 0, 5,4, 4, 0, 0, 0, 0, 5],
           [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3, 0, 3],
           [0, 0, 0, 0, 0, 0, 0, 4, 3, 0, 1, 0, 4, 0, 0, 4, 0, 0, 1, 0, 4, 0],
           [3, 3, 0, 0, 0, 0, 4, 0, 3, 1, 0, 4, 0, 4, 0, 3, 0, 0, 2, 2, 0, 0],
           [5, 5, 5, 0, 4, 0, 4, 0, 3, 0, 0, 0, 0, 0, 5, 0, 0, 0, 5, 5, 0, 0],
           [0, 0, 4, 0, 4, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 5, 0, 5, 0, 0, 5, 0],
           [4, 0, 4, 5, 0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 5],
           [0, 0, 4, 0, 4, 0, 0, 0, 0, 0, 0, 0, 5, 0, 4, 0, 4, 0, 0, 0, 0, 4],
           [0, 0, 0, 0, 0, 2, 3, 2, 5, 0, 0, 5, 2, 0, 2, 0, 2, 5, 0, 0, 5, 2],
           [0, 0, 0, 5, 0, 2, 0, 2, 5, 0, 5, 5, 2, 0, 0, 5, 0, 0, 0, 4, 5, 0],
           [1, 1, 0, 5, 0, 2, 0, 2, 5, 0, 0, 5, 2, 2, 1, 1, 2, 1, 0, 4, 5, 0],
           [2, 2, 0, 3, 2, 0, 0, 0, 4, 4, 0, 0, 0, 0, 5,4, 4, 0, 0, 0, 0, 5 ],
           [0, 2, 0, 3, 0, 4, 3, 0, 0, 2, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3, 0, 3],
           [0, 2, 0, 0, 0, 2, 0, 4, 0, 0, 1, 0, 4, 0, 0, 4, 0, 0, 1, 0, 4, 0],
           [3, 3, 0, 0, 0, 2, 4, 0, 0, 1, 0, 4, 5, 4, 0, 3, 0, 3, 2, 2, 0, 0],
           [5, 5, 5, 0, 4, 2, 4, 0, 0, 0, 0, 0, 5, 0, 5, 4, 3, 3, 5, 5, 0, 0],
           [0, 0, 4, 0, 4, 0, 0, 0, 5, 0, 3, 0, 5, 2, 0, 5, 0, 5, 0, 0, 5, 0],
           [4, 0, 4, 0, 0, 2, 0, 4, 5, 4, 3, 0, 0, 2, 0, 0, 0, 5, 0, 0, 0, 5],
           [0, 0, 4, 0, 4, 0, 3, 0, 0, 0, 3, 0, 5, 2, 4, 0, 4, 0, 1, 0, 0, 4],
           [0, 0, 0, 0, 0, 2, 3, 2, 5, 0, 3, 5, 2, 2, 2, 0, 2, 5, 1, 0, 5, 2],
           [0, 0, 0, 3, 2, 2, 0, 2, 5, 0, 3, 5, 2, 0, 0, 5, 0, 0, 1, 4, 5, 0],
           [1, 1, 1, 0, 0, 2, 0, 2, 5, 0, 3, 5, 2, 2, 1, 1, 2, 1, 0, 4, 5, 0],
           [2, 2, 0, 0, 2, 0, 0, 0, 4, 4, 0, 0, 0, 0, 5, 4, 4, 0, 3, 0, 0, 5],
           [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 0, 0, 3, 0, 4, 4, 0, 0, 0, 3, 0, 3],
           [5, 2, 4, 3, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0, 0, 4, 0, 0, 1, 0, 4, 0],
           [3, 3, 0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0, 4, 0, 3, 0, 0, 2, 2, 0, 0],
           [5, 5, 5, 0, 4, 0, 4, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 5, 5, 0, 0],
           [0, 1, 4, 0, 4, 0, 0, 1, 2, 3, 4, 5, 5, 0, 0, 5, 0, 5, 0, 0, 5, 0],
           [4, 2, 4, 0, 0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 5],
           [0, 3, 4, 0, 4, 0, 3, 5, 2, 2, 2, 0, 5, 0, 4, 0, 4, 0, 0, 0, 0, 4],
           [0, 3, 0, 0, 0, 2, 0, 2, 5, 0, 0, 5, 2, 0, 2, 0, 2, 5, 0, 0, 5, 2],
           [0, 5, 0, 0, 0, 2, 0, 2, 5, 0, 0, 5, 2, 0, 0, 5, 0, 0, 0, 4, 5, 0],
           [2, 2, 0, 3, 2, 0, 0, 0, 4, 4, 0, 0, 0, 0, 5,4, 4, 0, 0, 0, 0, 5 ],
           [0, 2, 0, 3, 0, 4, 3, 0, 0, 2, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3, 0, 3],
           [0, 2, 3, 5, 2, 2, 2, 4, 0, 0, 1, 0, 4, 0, 0, 4, 0, 0, 1, 0, 4, 0],
           [3, 3, 0, 0, 0, 2, 4, 0, 0, 1, 0, 4, 5, 4, 0, 3, 0, 3, 2, 2, 0, 0],
           [5, 5, 5, 0, 4, 2, 4, 1, 2, 4, 3, 4, 5, 0, 5, 4, 3, 3, 5, 5, 0, 0],
           [0, 0, 4, 0, 4, 0, 0, 0, 5, 0, 3, 0, 5, 2, 0, 5, 0, 5, 0, 0, 5, 0],
           [4, 0, 4, 3, 5, 2, 2, 2, 5, 4, 3, 0, 0, 2, 0, 0, 0, 5, 0, 0, 0, 5],
           [5, 0, 4, 0, 4, 2, 3, 0, 0, 0, 3, 0, 5, 2, 4, 0, 4, 0, 1, 0, 0, 4],
           [5, 4, 2, 0, 0, 2, 3, 2, 5, 0, 3, 5, 2, 2, 2, 0, 2, 5, 1, 0, 5, 2],
           [0, 0, 0, 3, 2, 3, 5, 2, 2, 2, 3, 5, 2, 0, 0, 5, 0, 0, 1, 4, 5, 0],
           [1, 1, 1, 0, 0, 2, 0, 2, 5, 0, 3, 5, 2, 2, 1, 1, 2, 1, 0, 4, 5, 0],
           [2, 2, 0, 0, 2, 0, 0, 3, 5, 2, 2, 2, 0, 0, 5, 4, 4, 0, 3, 0, 0, 5],
           [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 0, 0, 3, 0, 4, 4, 0, 0, 0, 3, 0, 3],
           [5, 2, 4, 3, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0, 0, 4, 2, 0, 1, 0, 4, 0],
           [3, 3, 0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0, 4, 0, 3, 4, 0, 2, 2, 0, 0],
           [5, 5, 5, 0, 4, 0, 4, 4, 0, 4, 4, 0, 4, 0, 5, 0, 5, 0, 5, 5, 0, 0],
           [0, 0, 4, 0, 4, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 5, 0, 5, 0, 0, 5, 0],
           [4, 0, 4, 0, 0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 5],
           [0, 0, 4, 0, 4, 0, 0, 0, 0, 0, 0, 0, 5, 0, 4, 0, 4, 0, 0, 0, 0, 4],
           [0, 0, 0, 0, 0, 2, 0, 2, 5, 0, 0, 5, 2, 0, 2, 0, 2, 5, 0, 0, 5, 2],
           [0, 0, 0, 0, 0, 2, 0, 2, 5, 0, 0, 5, 2, 0, 0, 5, 0, 0, 0, 4, 5, 0],           
           [1, 1, 0, 0, 0, 2, 0, 2, 5, 0, 0, 5, 2, 2, 1, 1, 2, 1, 0, 4, 5, 0]]
def ecludSim(inA,inB):
    """
    #通过欧氏距离计算相似度，并将相似度归一化到[0,1]之间
    sum(相减求平方)再开方
    """
    return 1./(1.0 + la.norm(inA - inB))

def pearsSim(inA,inB):
    """
    #皮尔逊相关系数归一化到[-1,-1]之间
    #度量两个向量的相似度
    #优点：对用户评级的量级不敏感，比如某个用户对所有物品评分都是5分，而另一个用户对所有物品评分都是1分
    #皮尔逊相关系数会认为这两个向量是相等的
    """
    if len(inA)<3:
        return 1.0
    return 0.5 + 0.5 *  np.corrcoef(inA, inB, rowvar = 0)[0][1] 
def cosSim(inA,inB):
    """
    #通过夹角余弦度量相似度并归一化到[-1,1]之间
    #如果两个向量夹角为90°则相似度为0，如果向量方向相等则相似度为1,相反则为-1
    """
    num = float(inA.T*inB) #向量内积，即数量积
    denom = la.norm(inA) * la.norm(inB) #向量A的模长*向量B的模长
    return 0.5 + 0.5*(num/denom)
@tim.timeit()  
def standEstimate(dataMat,user,simMeas,item):
    """
    #评分 = 总的评分/总的相似度
    #取出列(物品)的数量n,初始化总的相似度totalSim和总的评分totalSimRating
    #遍历每一列，如果用户的这一列为值UserRating =0(为0的属于未评分的)则跳过本次循环
    #否则取出整列中大于0的元素，与待评分的列中大于0的元素进行逻辑与操作，如果与操作长度为0则表示相似度为0，
    #与操作长度不为0,则计算计算余弦相似度similarity
    #累加相似度值:totalSim+=similarity
    #通过该列的评分乘以本次相似度值得到待评分列的预测值并累加:totalSimRating += similarity*UserRating
    #循环结束后如果总的相似度为0则返回0，否则返回 totalSimRating \ totalSim
    """
    columnCount = np.shape(dataMat)[1] #取列
    totalSimilarity = totalSimilarityRating = 0
    for j in range(columnCount):
        UserRating = dataMat[user,j]
        if UserRating == 0:
            continue
        OverLap = np.nonzero(np.logical_and(dataMat[:,j].A>0,dataMat[:,item].A>0))[0]
        if len(OverLap) == 0:
            similarity = 0
            continue
        else:
            similarity = cosSim(inA = dataMat[:,j], inB = dataMat[:,item])
        totalSimilarity += similarity
        totalSimilarityRating += similarity * UserRating
    if totalSimilarity == 0:
        return 0
    else:
        return (totalSimilarityRating / totalSimilarity)
def svdPercentage(sigma,percentage):
    """
    获取奇异值的前n个
    取奇异值的平方，通过累加前n个占比大于percentage时返回n
    """
    updateSum = 0
    sigma2 = [s**2 for s in sigma]
    totalSum = sum(sigma2)
    for k,v in enumerate(sigma2):
        updateSum += v
        if updateSum >= totalSum * percentage:
            return k
@tim.timeit()     
def svdEstimate(dataMat,transforedMat,user,simMeas,item,percentage=0.9):
    """
    通过svd将dataMat降维到低维空间，节省计算资源
    """
    
    columnCount = np.shape(dataMat)[1] #取列
    totalSimilarity = totalSimilarityRating = 0
    for j in range(columnCount):
        UserRating = dataMat[user,j]
        if UserRating == 0:
            continue
        similarity = cosSim(inA = transforedMat[j,:].T, inB = transforedMat[item,:].T)
        totalSimilarity += similarity
        #print("item %d item %d similarity is :%f" %(item,j,similarity))
        totalSimilarityRating += similarity * UserRating
    if totalSimilarity == 0:
        return 0
    else:
        return (totalSimilarityRating / totalSimilarity)
@tim.timeit()    
def recommandStand(dataMat,user,topN=3,simMeas = cosSim, estMethod = standEstimate):
    """
    #首先取出用户未评分的物品：取行中为0的item
    #对用户未评分物品进行遍历，通过相似度计算得出物品的评分,保存物品对应的评分
    #从高到低返回TopN的(物品，评分)
    """
    unRateItems = np.nonzero(dataMat[user,:].A == 0)[1]
    itemScores = []
    for i in unRateItems:
        score = standEstimate(dataMat=dataMat,user=user,simMeas=simMeas,item=i)
        itemScores.append((i,score))
        #print("user %d's  %dth item estimatedScore is %d" %(user,i,score))
    return sorted(itemScores,key=lambda x:x[0],reverse=True)[:topN]
@tim.timeit()    
def recommandSvd(dataMat, transforedMat, user, topN=3, simMeas = cosSim, estMethod = svdEstimate,percentage = 0.9):
    """
    #首先取出用户未评分的物品：取行中为0的item
    #对用户未评分物品进行遍历，通过相似度计算得出物品的评分,保存物品对应的评分
    #从高到低返回TopN的(物品，评分)
    """
    unRateItems = np.nonzero(dataMat[user,:].A == 0)[1]
    itemScores = []
    for i in unRateItems:
        score = svdEstimate(dataMat=dataMat,transforedMat = transforedMat,user=user,simMeas=simMeas,item=i)
        itemScores.append((i,score))
        #print("user %d's  %dth item estimatedScore is %d" %(user,i,score))
    return sorted(itemScores,key=lambda x:x[0],reverse=True)[:topN]    
    
if __name__ == '__main__':
    #myMat = np.mat(loadExData2())
    myMat = np.mat(np.random.randint(0,high=5,size=(1000,1000)))
    
    N = 3
    percentage = 0.8
    recommandList = recommandStand(myMat, user= 4,topN=N)
    U,Sigma,V = la.svd(myMat)
    k = svdPercentage(Sigma, percentage)
    sigK = np.mat(np.eye(k)*Sigma[:k])
    transforedMat = myMat.T * U[:,:k] *sigK.I
    
    recommandList = recommandSvd(myMat,transforedMat, user= 2,topN=N)
    print("Top:%d" % N)
    print(recommandList)
    Timing.show_timing_log()
    """
    Top:3
[(996, 2.5392431018339123), (993, 2.5394899191345037), (978, 2.5379055263547703)]

==============================================================================================================
Timing log
--------------------------------------------------------------------------------------------------------------
                                   [Method]                  svdEstimate :      24.99143 s (Call Time:    191)
                                   [Method]                 recommandSvd :      24.99643 s (Call Time:      1)
                                   [Method]                standEstimate :      51.58395 s (Call Time:    205)
                                   [Method]               recommandStand :      51.59195 s (Call Time:      1)
--------------------------------------------------------------------------------------------------------------
    
    """