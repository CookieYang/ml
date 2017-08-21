from numpy import *
def splitDataSet(dataSet1,dataSet2,test_proportion = 0.2, circle = 100):
    m = shape(dataSet1)[0]
    tm = m - int(test_proportion * m)
    c = dataSet1[0:tm,:]
    return dataSet1[0:tm,:],dataSet2[0:tm,:]