#  -*- coding: utf-8 -*-
import threading
import time
import numpy as np
from weightedKnnPredictorOnline import readDataFromCSV, getTrainingDataSetOnline, trainKnn, testKnnOnePoint, abnormalDetection, plotDataHist


#**************常量和全局变量可以在prepare方法中声明/定义*****************************
def prepare():
    #**************全局变量定义【可以直接在函数中声明为global后使用，但是为清晰起见，可以预先标出所有全局变量】*****************
    #存储原始数据序列 [（datetime, value）]
    global rawDataList
    rawDataList = [] #缓存队列，最大长度为 maxBufferSize
    
    #模型是否准备好（一次训练，连续在线预测），全局参数
    global modelReady
    modelReady = False
    
    #模型初始参数设置
    global featureLenDefault, maxKForModel
    featureLenDefault = 8 #特征长度，设置为8
    maxKForModel = 10 #最大K值设为10
    
    #模型训练得到的参数，全局变量
    global optimalInputSet, optimalK, optimalErrorList
    optimalInputSet = [] #最优特征集合flag, 0-1向量
    optimalK = 1 #最优K值，初始化为1
    optimalErrorList = [] #用于检测异常的相对误差队列，长度限制为testSizeThreshold（用于预测下一个值的样本集大小）

    #"常量"定义，Python中没有常量的概念，可以定义为全局的
    global trainingSizeThreshold, maxBufferSize, testSizeThreshold
    trainingSizeThreshold = 24*60*6/12/5 #取一天中的前2个小时的数据（720个点，10s为单位; 除以5后只有0.4小时）用来训练模型。这个阈值用于启动模型训练
    maxBufferSize = 24*60*6*7 #最大的缓存（元素个数）：保存一周的数据
    testSizeThreshold = 24*60*6/6 #用于(预测)的样本点不超过4h（1440个样本点）。 这个阈值用于限制样本集的大小，降低预测时间



    
#storm在线处理组件的核心函数，所有的操作都在这个函数中完成。
#每当有数据indata到来时，就会调用exec函数，没有数据到来时，会有其他函数处理（模拟时需要设计这种函数，。例如多线程等待）
#indata格式[datetime, value]list形式
def onlineAbnormalDetection(indata):
    #所有进来的数据都先缓存
    global rawDataList, maxBufferSize
    if indata is None:
        print 'No Data!'
        return (None, None) #返回空，保证函数的输出格式一致

    else: #数据不为空
        rawDataList.append(indata) #(datetime, value)原始数据
    #print 'rawDataList is: ', rawDataList
    
    #如果超过最大容量，则限制其大小，防止内存溢出。
    if len(rawDataList) > maxBufferSize:
        rawDataList.pop(0) #控制rawDataList的大小
    
    #如果判断模型是否准备好
    global modelReady, trainingSizeThreshold, featureLenDefault #全局变量(模型是否准备好，训练集大小，默认特征长度)
    if modelReady == False: #模型没准备好，需要先训练模型
        #训练模型首先要训练集足够，即rawDataList长度足够
        if len(rawDataList) < trainingSizeThreshold: #训练集长度不够
            return (None, None) #返回空，保证函数的输出格式一致
        else: #训练集长度达到，开始训练
            print 'training size is big enough, starting training...'
            trainingStartTime = time.clock() #训练开始时间
            trainingData = np.array(rawDataList)[:,1] #序列没有排序，默认每次读取的数据都是按时间先到到达的。 #取第二列value 
            #需要先将数据归一化后再训练，否则计算距离时，dist太大导致高斯权重为0，无法计算加权KNN值
            trainingDataNormalized = (trainingData-np.min(trainingData))*1.0/(np.max(trainingData)-np.min(trainingData))
            #构造训练集(归一化后的)
            trainingSetList = getTrainingDataSetOnline(trainingDataNormalized, featureLen=featureLenDefault) 
            
            #训练模型
            global optimalInputSet, optimalK, optimalErrorList #全局变量
            #训练返回最优特征值、最优K值，误差列表（初始长度300，因为训练中利用3次CV，每次validation有100个样本点/误差）
            optimalInputSet, optimalK, optimalErrorList = trainKnn(trainingSetList, featureLen=featureLenDefault, maxK=maxKForModel) #最大K值为10
            #训练误差反归一化(成为绝对误差)
            optimalErrorList = [item*(np.max(trainingData)-np.min(trainingData)) for item in optimalErrorList]         
                    
            
            trainingEndTime = time.clock() #训练结束时间
            
            print 'Model training is finished.'            
            print 'Model parameters: %s, %s' %(optimalInputSet, optimalK)
            print 'Model Training Error(Min, Average, Max): (%s, %s, %s)' %(np.min(optimalErrorList), np.mean(optimalErrorList), np.max(optimalErrorList))
            print 'total training time: %s minutes.\n' %((trainingEndTime-trainingStartTime)*1.0/60)
            
            
            #训练完毕，将标志位置为真            
            modelReady = True
            #训练结束，释放不需要的内存（因为函数死循环，所以尽可能释放不需要的内存，以免这些内存被无谓的长期占用）
            del trainingData
            del trainingDataNormalized
            del trainingSetList
            
            return (None, None) #返回空，保证函数的输出格式一致
            
    else: #模型已准备好，可以直接预测
        # 预测时不需要rawDataList这么大的数据集，只需要取testSizeThreshold（半天的数据集）用于预测即可
        global testSizeThreshold #限制用于预测的“训练集”大小
        #当模型刚训练好时，dataSetForTest大小只比训练集多一个元素，随着时间推移，dataSetForTest也要及时更新
        if len(rawDataList) < testSizeThreshold: #如果原始队列rawDataList中的数据量没有达到用于测试的集合大小限制，则取全部的rawDataList
            dataSetForTest = np.array(rawDataList)[:,1] #取第二列value
        else: #超过大小限制，则要动态更新dataSetForTest
            dataSetForTest = np.array(rawDataList[-testSizeThreshold:])[:,1] #取最后testSizeThreshold长度的数据用于预测，取第二列value
        #归一化（否则会造成距离太大，高斯权重为0，无法计算）
        dataSetForTestNormalized = (dataSetForTest-np.min(dataSetForTest))*1.0/(np.max(dataSetForTest)-np.min(dataSetForTest))
        
        #构造用于预测的数据集(归一化之后的)
        dataSetForTestList = getTrainingDataSetOnline(dataSetForTestNormalized, featureLen=featureLenDefault) #特征长度为8
        #最后一个元素是将要被预测的数据，前面N-1个用于预测最后一个元素
        dataSetForTestList = dataSetForTestList[:-1] #前N-1个用于预测
        testPoint = dataSetForTestList[-1] #最后一个元素将要被预测
        
        #单点预测，返回 np.array((原始值，预测值)) 都是归一化的值
        originalPredictedValue = testKnnOnePoint(testPoint, dataSetForTestList, optimalInputSet, optimalK)
        #预测结果反归一化
        originalPredictedValue = originalPredictedValue*(np.max(dataSetForTest)-np.min(dataSetForTest))
        
        #异常检测        
        attack = abnormalDetection(optimalErrorList, originalPredictedValue)
        
        #更新optimalErrorList，并限制其长度不超过testSizeThreshold
        optimalErrorList.append(originalPredictedValue[0]-originalPredictedValue[1]) #将新误差加入误差列表
        if len(optimalErrorList) > testSizeThreshold: #误差列表长度超过阈值，删掉旧值
            optimalErrorList.pop(0)
        
        #需要测试一下误差的分布情况        
        plotDataHist(np.array(optimalErrorList), figname='figure/predictionErrorDist.jpg') #只显示，不保存
        time.sleep(100)
        #print "At time %s, attack is %s" %(indata[0], attack)
        
        #输出（datetime, attack标记）
        return (indata[0], attack)


  
  
#模拟storm实时数据流，测试onlineAbnormalDetection(indata)这个在线数据处理组件的逻辑功能是否正确
def testStormComponent():
    #从文件中读取（datetime, value），存入list缓存中，这个相当于一个输入数据源。 
    #然后循环逐个从队列中读取元素，每次读取后调用onlineAbnormalDetection()函数，如果list中没有数据，则等待数据到来（sleep？ wait？）
    filename = 'data/dnsQuery10sDeleteMissingValue.csv'    
    dataList = readDataFromCSV(filename, flag='list')
    #print dataList
    
    #循环从list中读取数据，进行处理
    while(True):
        if len(dataList) > 0: #如果缓存队列中还有元素，就取出来处理
            indata = dataList.pop(0) #取出第一个元素 (datetime, value)
            #print 'element is: ', indata
            
            #实时处理，检测异常值，返回（时间、是否有攻击）
            returnValue = onlineAbnormalDetection(indata)   
            #print 'Detection Result:', returnValue
            if (returnValue != (None, None)) & (returnValue[1] != False): 
                dt = returnValue[0] #时间
                attack = returnValue[1] #是否有攻击的标志，True表示有攻击
                print "At time %s, attack is %s" %(dt, attack)
            
        else: #缓存队列中没有数据了
            time.sleep(5) #休眠等待5秒
    
    
    
if __name__ == "__main__":
    start = time.clock()
    print "Time started: ", time.strftime('%Y-%m-%d %H:%M:%S',time.localtime())

    #测试exec函数。设计输入（直接将本文读入到一个大list中，超过长度就删掉），exec函数从list中读取数据，进行处理
    prepare() #变量初始化（全局变量）
    testStormComponent() #测试组件功能
    
    
    end = time.clock()
    print "Time ended: ",  time.strftime('%Y-%m-%d %H:%M:%S',time.localtime())
    print "Time used: ", end-start, " seconds, i.e., ", (end-start)*1.0/60, "minutes"
