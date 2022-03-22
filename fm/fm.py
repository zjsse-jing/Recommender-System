'''
python实现fm的二分类
dataset: diabetes.csv diabetes皮马人糖尿病数据集

Factorization Machines Model，FM
'''
import numpy as np
import random
import pandas as pd
from random import normalvariate
from datetime import date, datetime
from numpy import mat,shape,multiply,log,zeros,ones

def load_data(filename, ratio):
    trainData = []
    testData = []
    with open(filename) as txt:
        lines = txt.readlines()
        for line in lines:
            lineData = line.strip().split(',')
            if random.random() < ratio:
                trainData.append(lineData)
            else:
                testData.append(lineData)
    np.savetxt('/content/sample_data/train_diabetes.csv', trainData, delimiter=',',fmt='%s')
    np.savetxt('/content/sample_data/test_diabetes.csv', testData, delimiter=',',fmt='%s')


def preprocessData(data):
    feature = np.array(data.iloc[:,:-1]) #8个特征
    label = data.iloc[:,-1].map(lambda x:1 if x==1 else -1)

    #数组归一化
    zmax,zmin = feature.max(axis=0), feature.min(axis=0)
    feature = (feature-zmin) / (zmax - zmin)
    label = np.array(label)

    return feature, label

def sigmoid(inx):
    return 1.0 / ( 1 + np.exp(-inx));

#损失函数
def getLoss(predict, classLabels):
    m = len(predict)
    loss = 0.0
    for i in range(m):
        loss -= log(sigmoid(predict[i] * classLabels[i]))
    return loss

#预测
def getPrediction(dataMatrix, w_0, w, v):
    m = np.shape(dataMatrix)[0]
    result = []
    for x in range(m):
        inter_1 = dataMatrix[x] * v
        inter_2 = multiply(dataMatrix[x], dataMatrix[x]) * multiply(v,v)
        interaction = np.sum(multiply(inter_1, inter_1) - inter_2) / 2.
        p = w_0 + dataMatrix[x] * w + interaction
        pre = sigmoid(p[0, 0])
        result.append(pre)
    return result

#准确率
def getAccuracy(predict, classLabels):
    m = len(predict)
    allItem = 0
    error = 0
    for i in range(m):
        allItem += 1
        if float(predict[i]) < 0.5 and classLabels==1.0:
            error += 1
        elif float(predict[i]) >= 0.5 and classLabels==-1.0:
            error += 1
        else:
            continue
    return float(error) / allItem

def FM(dataMatrix, classLabels, k, iter, alpha):
    '''
    :param dataMatrix: 特征矩阵
    :param classLabels: 标签矩阵
    :param k:           v的维数
    :param iter:        迭代次数
    return:             常数项w_0, 一阶特征系数w，二阶交叉特征系数v
    '''
    m, n = np.shape(dataMatrix) #矩阵行列数，样本数m，特征数n  (606, 8)

    #参数初始化
    w_0 = 0
    w = zeros((n,1))
    v = normalvariate(0, 0.2) * ones((n, k)) #辅助向量（n*K），训练二阶交叉特征系数

    for it in range(iter):
        for x in range(m): #随机优化，每次只取一个样本
            #二阶项计算
            inter_1 = dataMatrix[x] * v #样本(1*n)(n*k),得到k维向量
            inter_2 = multiply(dataMatrix[x], dataMatrix[x]) *  multiply(v,v)
            interaction = sum(multiply(inter_1, inter_1) - inter_2) / 2.

            p = w_0 + dataMatrix[x] * w + interaction # FM的全部项之和
            
            tmp = 1 - sigmoid(classLabels[x] * p[0, 0])
            # FM二分类随机梯度下降法训练 w_0 
            w_0 = w_0 + alpha * tmp * classLabels[x]

            for i in range(n):
                if dataMatrix[x, i] != 0:
                    w[i,0]=w[i,0] + alpha * tmp * classLabels[x] * dataMatrix[x, i]
                    for j in range(k):
                        #FM二分类 v(i,f) 
                        v[i,j] = v[i,j] + alpha * tmp * classLabels[x] * (dataMatrix[x, i] * inter_1[0,j]-v[i,j]*dataMatrix[x,i]*dataMatrix[x,i])
        if it % 10 == 0:
            loss = getLoss(getPrediction(mat(dataMatrix), w_0, w, v), classLabels)
            print("第{}次迭代后的损失为{}".format(it, loss))
    return w_0, w, v

if __name__ == '__main__':
    load_data('/content/sample_data/diabetes.csv', 0.8)
    trainData = pd.read_csv('/content/sample_data/train_diabetes.csv')
    testData = pd.read_csv('/content/sample_data/test_diabetes.csv')
    featTrain, labelTrain = preprocessData(trainData)
    featTest, labelTest = preprocessData(testData)
    date_train = datetime.now()

    w_0, w, v = FM(np.mat(featTrain), labelTrain, 4, 100, 0.01)
    print(w_0)
    print(w)
    print(v)

    predict_train_res = getPrediction(mat(featTrain), w_0, w, v)
    train_acc= getAccuracy(1 - predict_train_res, labelTrain)

    predict_test_res = getPrediction(mat(featTest), w_0, w, v)
    test_acc= getAccuracy(1 - predict_test_res, labelTest)


