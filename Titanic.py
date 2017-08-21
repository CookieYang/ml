import tensorflow as tf
import pandas as pd
import bmTools
tData = pd.read_csv(r'D:\datas\train.csv')
tData['Sex'] = tData['Sex'].apply(lambda s:1 if s == 'male' else 0)
tData = tData.fillna(0)
dataSetX = tData[['Sex','Age','Pclass','SibSp','Parch','Fare']]
dataSetX = dataSetX.as_matrix()
tData['Deceased'] = tData['Survived'].apply(lambda s: int(not s))
dataSetY = tData[['Survived','Deceased']]
dataSetY = dataSetY.as_matrix()
bmTools.splitDataSet(dataSetX,dataSetY)

