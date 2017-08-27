import bmTools as bt
import pandas as pd
import numpy as np
import bmTools as bm
def sigmod(theta, x):
    newMat =  - x * theta
    return 1 / (1 + np.exp(newMat))

def lossFuc(label, x, target):
    m = np.shape(x)[0]
    loss = 0
    for i in range(m):
        loss += float(label[i][0]) * np.log(float(target[i][0])) + (1 - float(label[i][0])) * np.log( 1 - float(target[i][0]))
    return - 1 / m * loss

def autoNorm(dataSet):
    minVals = np.min(dataSet,axis=0)
    maxVals = np.max(dataSet,axis=0)
    ranges = maxVals - minVals
    n = np.zeros(np.shape(dataSet))
    m = np.shape(dataSet)[0]
    n = dataSet - np.tile(minVals,(m,1))
    n = n / np.tile(ranges,(m,1))
    return n

orignalData = pd.read_csv(r'D:\datas\train.csv')
orignalData['Sex'] = orignalData['Sex'].apply(lambda s:1 if 'male' == s else 0)
orignalData = orignalData.fillna(0)
orignalX = orignalData[['Sex','Age','Pclass','SibSp','Parch','Fare']]
orignalX = np.matrix(orignalX)
orignalX = autoNorm(orignalX)
orignalX = np.column_stack((orignalX,np.ones(np.shape(orignalX)[0])))
orignalY = np.matrix(orignalData[['Survived']])
x_test, y_test, x_train, y_train = bm.splitDataSet(orignalX,orignalY)
theta = np.ones(np.shape(orignalX)[1])
theta = np.mat(theta).T

# train
Hx = sigmod(theta, x_train)
loss = lossFuc(y_train,x_train,Hx)
for i in range(300):
    newH = sigmod(theta, x_train)
    theta = theta - 0.01 * x_train.T * (newH - y_train)
    newLoss = lossFuc(y_train, x_train, newH)
    # if abs(loss - newLoss) < 0.001:
    #     break
    # loss = newLoss
# print(theta)
result = list(sigmod(theta, x_test))
result = list(map(lambda x:1 if x >=0.5 else 0,result))
accuracy = 0
for i in range(len(result)):
    if result[i] == y_test[i]:
        accuracy += 1
print('accuracy is %.9f'% (accuracy / len(result)))





