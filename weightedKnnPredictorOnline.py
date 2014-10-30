#  -*- coding: utf-8 -*-

import os
from math import * #导入数学运算，阶乘,开方，E值等
import matplotlib.pyplot as plt
import numpy as np #np.roots()可以求高次方程的根
import time #计算程序运行时间
import csv #处理CSV文件
import datetime #处理datetime类型
from scipy import stats  #数据拟合 PDF估计 KS——test


#************************************以下函数用于数据预处理*************************************************
# readDataFromCSV
# readDNSQeuryDataFromCSV
# getDataSet
#**********************************************************************************************************  
#从文件中读取array,并按照第一列排序。这里针对DNS日志数据，函数不具有普适性
def readDataFromCSV(filename, flag='array'):
    dataList = []
    with open(filename, 'rb') as csvfile:
        reader = csv.reader(csvfile) #逐行读取
        try:
            for line in reader:
                #格式转换，（datetime， int）
                dataList.append( [datetime.datetime.strptime(line[0], "%Y-%m-%d %H:%M:%S"), int(float(line[1]))] )
        except csv.Error as e: #捕捉异常
            sys.exit('file %s, line %d: %s' % (filename, reader.line_num, e))    
    dataList.sort(key=lambda x:x[0]) #按照第1个关键字排序，即按照时间排序，可以再返回值上做
    print "length of datalist:", len(dataList)   
    
    if flag == 'array':
        return np.array(dataList)  #转化为矩阵
    else:   #flag == 'list'
        return dataList #直接返回list
        
        
        
#从CSV文件读取DNS访问日志数据（并归一化？？）
def readDNSQeuryDataFromCSV(filename="data/dnslog3week_1min_filled.csv"):  
    #从文件获取datetime/queryCounts by 1min,按时间排序
    dnsQueryWithTime_1min = readDataFromCSV(filename)
    #截取第二列，DNS请求次数，用numpy表示
    dnsQuery_1min = dnsQueryWithTime_1min[:,1]
    
    #归一化
    #dnsQuery_1min = (dnsQuery_1min - np.min(dnsQuery_1min))*1.0/(np.max(dnsQuery_1min) - np.min(dnsQuery_1min))
    return dnsQuery_1min
    
    
#输入：dataArray -- 时间序列arrangement
#      testSetSize -- 测试集大小
#      featureLength -- 特征长度
#输出： 广义训练集、测试集
def getTrainingDataSet(dataArray, testSetSize, featureLength=60):
    #广义训练集、测试集的初始化
    trainingSetList = list()
    testSetList = list()
    
    #样本（X,Y）总数=序列总长度-特征长度
    totalSize = len(dataArray) - featureLength 
    #testSetSize = 31      
    trainingSetSize = totalSize - testSetSize  #获取广义训练集（包含验证集合）大小
    
    #构造（广义）训练集，用于训练和验证模型（交叉验证法）
    for i in range(trainingSetSize): #注意dataArray下标从0开始
        inputVec = dataArray[i : i+featureLength] 
        traffic = dataArray[i+featureLength] 
        trainingSetList.append( (inputVec, traffic) ) 
    
    #构造测试集，用于评价模型的准确度    
    for i in range(testSetSize): #注意dataArray下标从0开始
        inputVec = dataArray[i+trainingSetSize : i+trainingSetSize+featureLength] 
        traffic = dataArray[i+trainingSetSize+featureLength]     
        testSetList.append( (inputVec, traffic) ) 
    
    return trainingSetList, testSetList



#构造用于训练KNN的训练集（实时数据累积得到，只有训练集，没有测试集）
#输入：dataArray -- 时间序列array
#      featureLength -- 特征长度
#输出： 广义训练集
def getTrainingDataSetOnline(dataArray, featureLen=8):
    #广义训练集的初始化
    trainingSetList = list()
    
    #训练样本（X,Y）总数 = 序列总长度-特征长度
    trainingSetSize = len(dataArray) - featureLen 
    
    #构造（广义）训练集，用于训练和验证模型（交叉验证法）
    for i in range(trainingSetSize): #注意dataArray下标从0开始
        inputVec = dataArray[i : i+featureLen] 
        traffic = dataArray[i+featureLen] 
        trainingSetList.append( (inputVec, traffic) ) 
    
    return trainingSetList  #只构造训练集【测试数据是动态到达，每隔一定的间隔】



   
#************************************以下函数用于KNN预测器**************************************************
# euclidean
# getDistances
# gaussian
# weigthedKnnPredictor
#**********************************************************************************************************    
#计算两个向量np.array之间的欧几里得距离，flag为选取的vec特征值，即某几位被选用，通过遗传算法得到
#flag向量的长度与vec长度一致，表示vec某些位被选中或者未被选中，例如flag=np.array([1 0 1 ... 0 0 1])
def euclidean1(v1, v2, flag):
    if len(v1)!=len(v2) or len(v1)!=len(flag):
        print 'array dimension not consistent error!'        
        return None
    d = 0.0
    for i in range(len(v1)):  
        d += ( float(v1[i]-v2[i])*float(flag[i]) )**2
    return sqrt(d)   


#计算两个向量np.array之间的欧几里得距离，flag为选取的vec特征值，flag=np.array([1 0 1 ... 0 0 1])
#比上面函数的表达式更简单
def euclidean2(v1, v2, flag):
    #不是最高效的做法，可以先用flag掩码，将v1,v2矩阵缩短
#    d = np.dot( (v1-v2)*flag, (v1-v2)*flag ) #直接矩阵运算
#    return sqrt(d)
    
    #1.2K样本点的预测，从22秒将为13秒，效果十分显著！！！
    #防错处理    
    if len(v1)!=len(v2) or len(v1)!=len(flag):
        print 'array dimension not consistent error!'        
        return None
    #先将flag的0、1矩阵变为bool矩阵
    flag = np.array(flag, dtype=bool)
    #将v1 v2掩码，降低V1 V2的维数，这也是特征选择的最终目的，有些列（flag对应为0）是不用计算的
    v1 = v1[flag]
    v2 = v2[flag]
    #用缩短后的矩阵进行运算，降低运算复杂度
    d = np.dot(v1-v2, v1-v2)
    return sqrt(d)

	
#计算两个向量的互相关，flag为选取的vec特征值，flag=np.array([1 0 1 ... 0 0 1])
#因为聚类使用互相关为度量函数时，聚类效果更好，这里为各类建模，是否也应该使用互相关？？？？？
def crossCorrelation(v1, v2, flag):
    v1 = v1*flag
    v2 = v2*flag
    return np.dot( (v1-np.mean(v1))*1.0/np.std(v1), (v2-np.mean(v2))*1.0/np.std(v2) )/len(v1)
   
    
#借助距离函数（欧几里得距离），计算某个向量与数据集中所有向量的距离，返回距离列表，后文KNN预测做准备。
#输入：trainingDataSetList —— 某类中所有小区训练集构成的list，格式为[(np.array(input features), trafficLoad)]
#      inputVec —— 表示某个输入特征向量np.array(input features)，不包含最后的trafficload值
#      flag —— 指示input set的数组，长度与vec长度一致
def getDistances(trainingDataSetList, inputVec, flag):
    distanceslist = [] #(index, dist)元组，表示第向量与data集中的第index向量的距离
    for i in range(len(trainingDataSetList)):
        vecTmp = trainingDataSetList[i][0] #从训练集中取出一个记录，并取出对应的输入向量np.array(input features)
        #距离采用互相关度量
        distanceslist.append( (euclidean2(inputVec, vecTmp, flag), i) ) #flag用于特征选择，欧氏距离
        #distanceslist.append( (crossCorrelation(inputVec, vecTmp, flag), i) ) #flag用于特征选择
    distanceslist.sort() #根据距离进行排序，方便kNN预测时选取K个近邻
    return distanceslist
        
        

# 加权KNN中预测时各result值的权重值，选择高斯权重，注意sigma参数为自定义的，可能影响
#最后的预测效果
def gaussian(dist, sigma=10.0):
    return e**(-dist**2/(2*sigma**2))
    
 
   
#加权KNN预测器，最优k值选取要通过cross-validation确定，这里默认是3
#本KNN预测器只预测某个向量对应的值，即只预测一个值！！！
#输入参数： trainingDataSetList——训练集构成的list，格式为[(np.array(input features), Y)]
#          inputVec——被预测值的输入属性向量
#          flag —— 指示输入集选择的标志位数组，长度与vec一致，例如flag=np.array([1 0 1 ... 0 0 1])
#          k——KNN中的K值，在使用训练集 CV确定模型时，这里的k值取一定范围，
#             根据交叉验证误差，选取最优的k值。确定最优k值后，在测试集上评估模型误差
#             时，k保持不变，即固定为最优模型中的k值。
#         weightf——加权KNN预测器的权重值，这里默认为高斯权重     
def weigthedKnnPredictor(trainingDataSetList, inputVec, flag, k, weightf=gaussian):
    #得到距离list
    dlist = getDistances(trainingDataSetList, inputVec, flag)
    predictedValue = 0.0
    totalWeight = 0.0
    
    for i in range(k):
        dist = dlist[i][0] #获取本向量与前k个近邻的距离
        idx = dlist[i][1] ##获取本向量与前k个近邻向量的下标
        weight = weightf(dist) #KNN加权，采用高斯权重
        #print 'weight:', weight
        predictedValue += weight*trainingDataSetList[idx][1] #前k个近邻对应的trafficLoad值
        totalWeight += weight
    predictedValue = predictedValue/totalWeight
    return predictedValue
 
	

#************************************以下函数用于KNN模型训练和测试******************************************
# holdout(CV)
# inputSelection (穷举搜索best input set，适合特征长度较小的时候)
# geneticAlgorithm（次优搜索best input set，适合特征长度较大的时候）。两种方法都是wrapper方法。还有MI等
# trainKnn
# testKnn
#**********************************************************************************************************  
#CV的实现，用Holdout方式，训练集中，随机选择100个作为验证数据，其余的为训练数据
#flag表示特征选择的bool向量
#k为KNN模型中的K值
#TrainingSet为训练集list
def holdout( flag, k, TrainingSet, algf=weigthedKnnPredictor):   
    trials = 3 # 3-FOLD CV
    totalError = 0
    errorList = [] #用来记录训练误差，为后面异常检测准备
    holdOutNum = 100 #holdOutNum留出的验证集
    for i in range(trials):
        #构造训练集、验证集
        #随机选取holdOutNum个数据为验证集，剩下的为训练集    
        ValidationSetList = []
        TrainingSetList = list(TrainingSet) #初始训练集为全集
        
        for i in range(holdOutNum):
            #保证每次抽取的数据不相同  
            index = np.random.randint(0, len(TrainingSetList)) #从训练集中随机选择一个
            ValidationSetList.append( TrainingSetList[index] ) #加入验证集
            popped = TrainingSetList.pop( index ) #从训练集中删除
    
        #一次hold-out测试求误差    
        trialError = 0 #一次实验的误差
        for i in range(len(ValidationSetList)): #预留某个小区的数据做验证集
            inputVec = ValidationSetList[i][0] #输入特征向量
            output = ValidationSetList[i][1]   #对应的输出值
            guess = algf(TrainingSetList, inputVec, flag, k) #使用weighted-KNN模型进行预测
            trialError += (output - guess)**2
            errorList.append(output-guess) #训练误差，y-yHat
        trialError = trialError*1.0/len(TrainingSetList)
        
        totalError += trialError
    return (totalError*1.0/trials, errorList) #返回训练总误差、 误差list
    


#由于输入特征长度为8，直接遍历所有可能组合即可，这里指定KNN中的K值
def inputSelection(k, TrainingSet, genLen, costf=holdout, algf=weigthedKnnPredictor):
    #穷举构造所有的输入组合 2**len -1 种
    inputList = [  ] 
    #print type(genLen)
    
    #构造所有的特征组合
    total = 2**genLen #int( pow(2, genLen) ) #2**genLen #总数
    for i in range(1, total): #从1--total-1
        binary = list(bin(i))[2:] #将i转化为2进制 '0b xxxx'，然后转为list，剔除前缀'0b'
        #根据位数，将总长度补全为genLen
        item = [0]*(genLen-len(binary)) + [int(v) for v in binary] #字符转化为int，在前面补零
        inputList.append(item)
    print "Number of candicate inputs: ", len(inputList)

    #计算每种特征组合情况下的代价  
    scores = [( costf(v, k, TrainingSet),  v ) for v in inputList] #（训练总误差、误差list、输入特征向量）list
    scores.sort()  #按照训练误差大小排序
    
    inputSet = np.array(scores[0][1]) #指定K值下的最优的个体，即对应的输入集，list -> np.array
    cost = scores[0][0][0]  #指定K值下的最优输入集对应的代价（CV误差）
    errorList = scores[0][0][1] #训练误差list，用于后续异常检测
    print 'best individual:', inputSet, cost
    return inputSet, cost, errorList #返回指定K值情况下，最优输入特征集和对应的代价（CV验证误差），以及对应的验证误差list
    
    


def trainKnn(trainingSetList, featureLen, maxK=10): #maxK=10默认值
    #初始化结果list,格式为[(cost, k, inputSet)] 三元组，将cost排在最前，方便list按照个体的cost排序，代价越小越优秀
    results = list()
    
    for k in range(1, maxK+1): #k取值从1到maxK
        #交叉验证的复杂度太高，改用hold out测试
        print 'Number of nearest neighors: %d' %k
        
        #全局搜索方法选择最优特征和K值，不适用与特征数较多的情况
        inputSet, cost, errorList = inputSelection(k, trainingSetList, featureLen, holdout, weigthedKnnPredictor)
        
        #利用遗传算法搜索次优的inputset k，适合特征数较多的情况，但是运行速度慢
#        inputSet, cost = geneticAlgorithm(k, trainingSetList, genLen=featureLen, costf=holdout, algf=weigthedKnnPredictor, \
#                                           popsize=50, mutprob=0.1, elite=0.2, maxiter=20) #特征长度
        results.append( (cost, k, inputSet, errorList) )
        print '*************************************************************'
        print '\n'
        
#    #排序之前绘图    
#    #绘制kNN中的K值变化对性能的影响，标注最佳（inputSet和K值）
#    plt.figure(figsize=(20,10))
#    index = range(1, maxK+1)
#    mse = [e for (e, k, featureSet) in results]
#    plt.plot( index, mse, 'r-*')
#    plt.title('MSE for kNN w.r.t k')
#    plt.xlabel('number of neighors (k)')
#    plt.ylabel('MSE for kNN validation')
#    plt.grid()
#    #plt.show()
#    plt.savefig("figure/Input_K_Selection.jpg")
    
    
    #默认按照元组的第一栏排序，即按照个体的cost排序，代价越小（越适应）的个体排在前面
    results.sort()
    
    #获取该类kNN模型的最优结构与参数：inputSet 和 K
    optimalK = results[0][1]
    optimalInputSet = results[0][2]
    optimalErrorList = results[0][3]
	
    print '\n optimal input set  -- optimal K'
    print optimalInputSet, optimalK
    print '\n'
    return optimalInputSet, optimalK, optimalErrorList #返回最佳输入特征集合、最优K值，及对应的误差list【用于异常检测】
    

   
#输入testPoint是（X,Y）元组，X为输入特征
#trainingSetList 是训练集list [(X, Y)]
def testKnnOnePoint(testPoint, trainingSetList, optimalInputSet, optimalK):   
    #array (originalValue, predictedValue) ] #用numpy矩阵比用list速度要快一点，内存占用也更少
    originalPredictedValue = np.zeros((1, 2)) #1行 2列，分别存储原始值、预测值
    
    #获取第i天的输入特征向量和对应的Y值   (X,Y)元组拆分
    inputVec = testPoint[0]
    originalValue = testPoint[1]
    
    #预测一个值，并存入原始预测值矩阵 np.array [一行两列，分别存储原始值、预测值]    
    predictedValue = weigthedKnnPredictor(trainingSetList, inputVec, optimalInputSet, optimalK)   
    originalPredictedValue = np.array([originalValue, predictedValue])
    
    #预测完之后将testPoint加入训练集，并限制训练集的大小 【24*60*10】一天有8640个样本点
    trainingSetList.append(testPoint) #测试样本点(X,Y)加入训练集
    if len(trainingSetList) > 24*60*60/10: #trainingSizeThreshold
        trainingSetList.pop(0) #删除第一个元素，保证训练集的大小
 
    return originalPredictedValue # array [(originalValue, predictedValue)]， 【1行2列】
    
      

    
#根据预测结果的正态分布，确定异常值
def abnormalDetection(errorList, originalPredictedValue):
    attack = False #首先默认没有攻击
    
    #本次预测误差   
    #error = originalPredictedValue[0,0] - originalPredictedValue[0,1] # y-yHat
    error = originalPredictedValue[0] - originalPredictedValue[1] # y-yHat
    
    #先将本次误差加入到误差列表中去，同时控制errorList的长度为1天
    errorList.append(error)
    if len(errorList) > 24*60*60/10:
        errorList.pop(0) #删除旧元素，保证errorList不要超长，否则会内存溢出
        
    #****************进行攻击检测************************    
    errorArray = np.array(errorList) #转化为np.array方便运算
    
    #误差上下界
    down = np.mean(errorArray) - 3*np.std(errorArray)
    up = np.mean(errorArray) + 3*np.std(errorArray)
    
    
    if (error <= down ) or (error >= up): #误差超过上下限
        attack = True #将攻击标志置为真
    
    return attack #返回（攻击标志）
    


#绘制直方图、用KDE估计密度曲线. 输入sample矩阵np.array
def plotDataHist(sample, figname=''):
    #画原始数据的分布图和拟合的分布图，并保存
    #利用直方图/KDE，画数据本身的PDF，也画原始数据hist图。但是bin的数目影响图的形状
    
    plt.figure()
    #直接画原始数据的直方图，形状与bin数目有关
    #根据样本数确定bin数目
    N = len(sample) #样本数目
    if 0 < N < 100:
        binNum = N
    elif 100 <= N < 2000:
        binNum = 100
    elif 2000 <= N < 6000:
        binNum = 200
    elif 6000 <= N < 9000:
        binNum = 300
    elif 9000 <= N < 15000:
        binNum = 400
    else:
        binNum = 600
    #绘制直方图
    plt.hist(sample, bins=binNum, normed=True, alpha=0.5, label='Data Histogram')
    plt.hold(True)
    
    #通过KDE画原始数据的PDF，图形与带宽有关。但是比利用histogram估计更加准确！
    xs = np.linspace(sample.min()-1, sample.max()+1, len(sample))#画图时x轴取值
    kde1 = stats.gaussian_kde(sample)
    plt.plot(xs, kde1(xs), 'r-', label="Measured PDF(KDE)")
    plt.hold(True)
    plt.grid()
    
#    plt.xlabel('MIPS')
#    plt.ylabel('density')
#    plt.title('MIPS Histogram')
    
    plt.xlabel('Relative Prediction Error')
    plt.ylabel('density')
    plt.title('Relative Prediction Error Histogram')
    plt.legend(loc='best', fancybox=True, shadow=False, prop={'size':12})
    
    #切换到figure文件夹目录，目录不存在则新建目录，为保存图片准备
    pwd = os.getcwd()
    fig_path = os.path.join(pwd, 'figure')
    if not os.path.isdir(fig_path):
        os.makedirs(fig_path)
        
    if figname != '': #只有在给定图片名时，才保存图片
        plt.savefig(figname, dpi=300)    
    
    


def testKnn(testSetList, trainingSetList, optimalInputSet, optimalK):
    steps = len(testSetList) #预测步数，即testSetList长度
    
    #array (originalValue, predictedValue) ] #用numpy矩阵比用list速度要快一点，内存占用也更少
    originalPredictedValueArray = np.zeros((steps, 2)) #steps行 2列，分别存储原始值、预测值
    
    #注意两重循环的次序，每次预测1min的数据
    for i in range(steps): 
        #获取第i天的输入特征向量和对应的Y值
        inputVec = testSetList[i][0]
        originalValue = testSetList[i][1]
            
        #预测值
        predictedValue = weigthedKnnPredictor(trainingSetList, inputVec, optimalInputSet, optimalK)
            
        #（原始值，预测值）元组插入list中
        #originalPredictedValueList.append( (originalValue, predictedValue) )
        originalPredictedValueArray[i] = np.array([originalValue, predictedValue])
            
        #一步/天预测完成后，更新训练集，将该小区第i天的测试集数据插入训练集中，作为后续预测是已知条件
        #*************注意：这一步会造成训练集的逐步扩大，影响计算性能。考虑固定窗口滑动预测又会损失精度************
        trainingSetList.append(testSetList[i]) #第i天的测试数据，加入训练集
        
        #固定窗口滑动预测【当训练集超过3天 3*24*60,则将就的训练样本点删除，保证训练集总大小为3天】
        if len(trainingSetList) > 3*24*60: #限制预测时的训练集不会超过3天的数据， 【3天足够，如果预测时间过长（超过10s），可以降为1天】
            trainingSetList.pop(0) #剔除最前面的元素，以保证set总大小的限制

    return originalPredictedValueArray # array [(originalValue, predictedValue)]
    
    
#时间序列的ACF分析
def acfPlotOfArray(data, key='hour', N=900, figureName = '' ): #900为自相关系数最大滞后的阶数
    acfDNS = np.zeros(N) #输出的ACF序列
    for i in range(N):
        acfDNS[i] = calAcfTimeSeries(data, i)
    #print acfDNS
    
    plt.figure(figsize=(20,10)) #默认dpi=80     
    plt.plot( np.arange(N), acfDNS, 'r-*') #DNS访问量的ACF
    #figureName = '' #解决local variable问题
    if key == 'min':
        #plt.title('ACF of DNSQuery_1min')
        plt.title('ACF of DNSQuery_1min Residual')
        plt.xlabel('Lag (min)', fontsize=11)
        
        ax = plt.gca()
        ticks = np.arange(0, N, 60)
        ax.xaxis.set_ticks(ticks) #设为以5min为周期
        #figureName = 'figure/AcfDNSQuery_1min.jpg'
        
    elif key == 'hour':
        plt.title('ACF of DNSQuery_1hour')
        plt.xlabel('Lag (hour)', fontsize=11)
        
        ax = plt.gca()
        ticks = np.arange(0, N, 12)
        ax.xaxis.set_ticks(ticks) #设为以12hour为周期
        #figureName = 'figure/AcfDNSQuery_1h.jpg'
    
    elif key == '10s':
        plt.title('ACF of DNSQuery_10s')
        plt.xlabel('Lag (*10s)', fontsize=11)
        
        ax = plt.gca()
        ticks = np.arange(0, N, 60*6*2)
        ax.xaxis.set_ticks(ticks) #设为以12hour为周期
        #figureName = 'figure/AcfDNSQuery_1h.jpg'
        
    plt.grid()
    plt.ylabel('Auto-correlation coefficient (ACF)', fontsize=11)
    plt.xlim(0, N)
    plt.ylim(-0.5, 1)

    #切换到figure文件夹目录，为保存图片准备
    pwd = os.getcwd()
    fig_path = os.path.join(pwd, 'figure')
    if not os.path.isdir(fig_path):
        os.makedirs(fig_path)
    #os.chdir(fig_path)    
    
    plt.savefig(figureName) #, dpi=300
    #plt.show()
    
    
    
#评价KNN模型的预测性能的指标（RMSE,MAPE）
#输入参数：
#testSet/testSetPredicted -- 原始值和测试值,numpy矩阵的形式，方便运算
#RMSE -- 均方根误差 sqrt(1/n * sum[ |e(i)|**2 ])
#MAPE -- 平均绝对百分比误差，默认选项   /n * sum[ |e/y(i)| ]
def PerformanceEval(testSet, testSetPredicted):    
    MAPE = (np.absolute(testSet-testSetPredicted)/testSet).sum()*1.0/len(testSet)
    print "MAPE for kNN predictor is :", MAPE
      
    RMSE = sqrt(((testSet - testSetPredicted)**2).sum()*1.0/len(testSet))
    print "RMSE for kNN predictor is :", RMSE

    index = np.arange(1, len(testSet)+1) 
    plt.figure(figsize=(20,10))
    plt.plot(index, testSet, color='b', linestyle='-', marker='*', label='Real Trace')
    plt.plot(index, testSetPredicted, color='r', linestyle='--', marker='d', label='Prediction')
    plt.legend(loc='best', fancybox=True, shadow=False, prop={'size':9})
    plt.xlabel('Time (min)', fontsize=9)
    plt.ylabel('DNS Query by Minute', fontsize=9)
    plt.title('Prediction Performance for DNS Query_1min')
    plt.xlim(1, len(testSet)+1)
    plt.grid()
      
    #切换到figure文件夹目录，为保存图片准备
    pwd = os.getcwd()
    fig_path = os.path.join(pwd, 'figure')
    if not os.path.isdir(fig_path):
        os.makedirs(fig_path)
          #os.chdir(fig_path) 
    
    figname = 'figure/predictedOriginalDnsQuery.jpg'
    plt.savefig(figname)
    
    
    #绘制残差序列
    residual = testSetPredicted - testSet #(yHat -y)
    index = np.arange(1, len(testSet)+1) 
    plt.figure(figsize=(20,10))
    plt.plot(index, residual, color='b', linestyle='-', marker='*')
    plt.xlabel('Time (min)', fontsize=9)
    plt.ylabel('DNS Query_1min Residual', fontsize=9)
    plt.title('DNS Query_1min Residual For KNN')
    plt.xlim(1, len(testSet)+1)
    plt.grid()
    
    figname = 'figure/predictionResidual.jpg'
    plt.savefig(figname)
    
    
    
    #绘制残差的ACF曲线图
    acfPlotOfArray(residual, key='min', N=900, figureName = 'figure/predictionResidualACF.jpg' )
    
    #return MAPE, RMSE

    

    
  
if __name__ == "__main__":
    start = time.clock()
    print "Time started: ", time.strftime('%Y-%m-%d %H:%M:%S',time.localtime())
    
    
#    #获取dns请求数目的时间序列    
#    dnsQuery3week_1min = readDNSQeuryDataFromCSV() #没有归一化的值
#    #归一化
#    valueRange = (np.max(dnsQuery3week_1min) - np.min(dnsQuery3week_1min))
#    minValue = np.min(dnsQuery3week_1min)
#    dnsQuery3week_1min_normalized = (dnsQuery3week_1min - minValue)*1.0/valueRange
#    print len(dnsQuery3week_1min)
#    
#    #输入参数特征长度：featureLen
#    featureLen = 8   #目前，特征长度为8时，为最优
#    
#    ##特征向量长度为60，即利用1h的数据作为输入。测试集长度为最后18天，即利用前3天训练模型，最后18天测试模型。    
#    trainingSet, testSet = getDataSet(dnsQuery3week_1min_normalized, 20*24*60, featureLength=featureLen) 
#    print 'total training set size:', len(trainingSet)
#    print 'total test set size:', len(testSet)
#    trainingSet_mini = trainingSet[0: int(0.15*len(trainingSet))] #
#    testSet_mini = trainingSet[int(0.15*len(trainingSet)):]
#    
#    print 'training set: ', len(trainingSet_mini)
#    print 'test set: ', len(testSet_mini)
#    
#    #由于是随机选择数据做验证集，因而每次运行产生的input和K都不一样。选择较优的即可。
#    #optimalInputSet, optimalK = trainKnn(trainingSet_mini, featureLen, maxK=15) 
#
#    #目前训练得到的模型最佳参数，为featureLen为8时
#    #特征长度为6时的最优特征和K
#    if featureLen == 6:
#        optimalInputSet = [0, 0, 1, 0, 1, 1]
#        optimalK = 10
#    
#    #特征长度为7时的最优特征和K值
#    elif featureLen == 7:
#        optimalInputSet = [0, 0, 0, 1, 1, 0, 1]
#        optimalK = 7
#    
#    #*******************目前最优参数**********************************
#    #特征长度为8时的最优特征和K值
#    elif featureLen == 8:        
#        optimalInputSet = [0, 0, 0, 0, 1, 0, 0, 1]
#        optimalK = 6
#    
#    #特征长度为9时的最优特征和K值
#    elif featureLen == 9:
#        optimalInputSet = [0, 0, 0, 0, 0, 1, 0, 0, 1]
#        optimalK = 7
#    
#    #特征长度为10时的最优特征和K值
#    elif featureLen == 10:
#        optimalInputSet = [0, 0, 1, 1, 0, 0, 1, 0, 0, 1]
#        optimalK = 5 
#
#    
#    # 测试KNN模型，输出[(originalValue, predictedValue) ]
#    OriginalPredictedValueArray = testKnn(testSet, trainingSet_mini, optimalInputSet, optimalK)
#
#    
#    originalValueArray = OriginalPredictedValueArray[:,0]
#    predictedValueArray = OriginalPredictedValueArray[:,1]
#    #反归一化
#    originalValueArray = valueRange*1.0*originalValueArray + minValue
#    predictedValueArray = valueRange*1.0*predictedValueArray + minValue
#
#    #评价模型的性能
#    PerformanceEval(originalValueArray, predictedValueArray)

        
    end = time.clock()
    print "Time ended: ",  time.strftime('%Y-%m-%d %H:%M:%S',time.localtime())
    print "Time used: ", end-start, " seconds"
