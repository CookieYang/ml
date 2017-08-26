import bmTools as bt
import pandas as pd
import numpy as np

def sigmod(theta, x):
    return 1 / 1 + np.exp(- theta.T * x)

orignalData = pd.read_csv(r'/Users/yangtao/Desktop/data/train.csv')
orignalData['Sex'] = orignalData['Sex'].apply(lambda s:1 if 'male' == s else 0)
orignalData = orignalData.fillna(0)
orignalX = orignalData[['Sex','Age','Pclass','SibSp','Parch','Fare']]
orignalX = np.matrix(orignalX)
orignalX = np.column_stack((orignalX,np.ones(np.shape(orignalX)[0])))
orignalData['Deceased'] = orignalData['Survived'].apply(lambda s: int(not s))
orignalY = np.matrix(orignalData[['Survived']])
theta = np.ones(np.shape(orignalX)[1])