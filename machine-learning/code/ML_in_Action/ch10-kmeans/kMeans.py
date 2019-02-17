from numpy import *
def loadDataSet(fileName):
    dataMat =[]
    with open(fileName,'r') as fr:
        for line in fr.readlines():
            curLine = line.strip().split('\t')
            fltLine = list(map(float, curLine)) #map all elements to float()
            dataMat.append(fltLine)
    return mat(dataMat)

def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB,2))) #la.norm(vecA-vecB)

def randCent(dataSet, k):
    """
    随机选取k个质心,n列
    对每个特征，取其最大最小值，得到范围区间
    设置质心为最大最小值之间
    """
    n = shape(dataSet)[1]
    centroids = mat(zeros((k,n))) #初始化质心矩阵,k,n 维，n表示特征数量
    for j in range(n):
        minJ = min(dataSet[:,j])
        rangJ = float(max(dataSet[:,j]) - minJ)
        centroids[:,j] = mat(minJ + rangJ * random.rand(k,1))
    return centroids
def kMeans(dataSet, k , distMeas=distEclud, createCent=randCent):
    """
    1.获取样本的个数
    2.初始化(m,2)的矩阵,第一维用于存放簇类别，第二维用于存放误差
    3.当任意一个样本的簇分配结果发生变化时
       对数据中的每个样本
           对每个质心
               计算质心与样本点之间的距离
           将数据点分配到距其最近的簇
        对每一个簇，计算簇中所有点的均值并将均值作为质心
           
    """
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m,2)))
    centroids = createCent(dataSet, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = inf; minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j,:],dataSet[i,:])
                if distJI < minDist:
                    minDist = distJI; minIndex = j
            if clusterAssment[i,0] != minIndex:
                clusterChanged = True
            clusterAssment[i,:] = minIndex,minDist**2
        for cent in range(k):
            ptsInClust = dataSet[nonzero(clusterAssment[:,0].A == cent)[0]]
            centroids[cent,:] = mean(ptsInClust,axis=0)
    return centroids,clusterAssment


def biKmeans(dataSet, k, distMeas=distEclud):
    """
    二分k均值算法
    将所有的点看成一个簇，
    当簇数目小于k时
      对于每一个簇
         计算总误差
         在给定的簇上面进行K-均值聚类（k=2)
         计算将该簇一分为二之后的总误差
      选择使误差最小的那个簇进行划分操作
    """
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m,2))) #(80,2)
    centroid0 = mean(dataSet, axis=0).tolist()[0]#取列方向的均值
    centList =[centroid0] #create a list with one centroid 
    bestClustAss = []
    for j in range(m):#calc initial Error
        clusterAssment[j,1] = distMeas(mat(centroid0), dataSet[j,:])**2 #全部为0类簇
    while (len(centList) < k):
        lowestSSE = inf
        for i in range(len(centList)): #对每个簇，计算其裂变后的优化误差，取优化误差最小的簇
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:,0].A==i)[0],:]#get the data points currently in cluster i
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)
            sseSplit = sum(splitClustAss[:,1])#compare the SSE to the currrent minimum
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:,0].A!=i)[0],1]) 
            print("sseSplit, and notSplit: ",sseSplit,sseNotSplit)
            if (sseSplit + sseNotSplit) < lowestSSE:
                bestCentToSplit = i        #簇的类别
                bestNewCents = centroidMat #质心赋值
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
        bestClustAss[nonzero(bestClustAss[:,0].A == 1)[0],0] = len(centList) #将优化误差最小的簇的簇类别由1分为2后(0,1)更新为(原始簇号，新增的簇号)，
        bestClustAss[nonzero(bestClustAss[:,0].A == 0)[0],0] = bestCentToSplit
        print('the bestCentToSplit is: ',bestCentToSplit)
        print('the len of bestClustAss is: ', len(bestClustAss)) #更新质心,裂变的簇的质心更新为裂变后的两个质心
        centList[bestCentToSplit] = bestNewCents[0,:].tolist()[0]#replace a centroid with two best centroids 
        centList.append(bestNewCents[1,:].tolist()[0]) #将产生裂变的簇的样本全部更新后裂变后所在的簇
        clusterAssment[nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:]= bestClustAss#reassign new clusters, and SSE
    return mat(centList), clusterAssment
     
def distSLC(vecA, vecB):
    a = sin(vecA[0,1]*pi/180) * sin(vecB[0,1]*pi/180)
    b = cos(vecA[0,1]*pi/180) * cos(vecB[0,1]*pi/180) * \
        cos(pi * (vecB[0,0]-vecA[0,0]) /180)
    return arccos(a + b)*6371.0 #pi is imported with numpy

import matplotlib
matplotlib.use('Agg') #savefig时需要设置'Agg'属性
import matplotlib.pyplot as plt


def clusterClubs(numClust = 5):
    datList = []
    for line in open("places.txt").readlines():
        lineArr = line.split('\t')
        datList.append([float(lineArr[4]), float(lineArr[3])])
        
    datMat = mat(datList)
    myCentroids, clustAssing = biKmeans(datMat, numClust, distMeas=distSLC)
    fig = plt.figure()
    
    rect =[0.1,0.1,0.8,0.8]
    scatterMarkers =['s','o','^','8','p','d','v','h','>','<']
    axprops = dict(xticks =[],yticks=[])
    ax0 = fig.add_axes(rect, label='ax0',**axprops)
    imgP = plt.imread('Portland.png')
    ax0.imshow(imgP)
    ax1=fig.add_axes(rect, label='ax1', frameon=False)
    for i in range(numClust): #对每个类别
        ptsInCurrCluster = datMat[nonzero(clustAssing[:,0].A == i)[0],:]
        markerStyle = scatterMarkers[i % len(scatterMarkers)]
        ax1.scatter(ptsInCurrCluster[:,0].flatten().A[0], ptsInCurrCluster[:,1].flatten().A[0], marker=markerStyle, s=90) #flatten()拉平操作
    ax1.scatter(myCentroids[:,0].flatten().A[0], myCentroids[:,1].flatten().A[0],marker="+", s=300)#绘制质心
    
    plt.show()
    plt.savefig('./fig_cat.png')
if __name__ =="__main__":
    dataMat = loadDataSet('testSet.txt')
    centroids = randCent(dataMat, 5)
    #kMeans(dataMat, 5)
    biKmeans(dataMat, 5)
    print(distEclud(centroids[0], centroids[1]))
    clusterClubs(5)