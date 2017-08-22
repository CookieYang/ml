import numpy as np
import random
def splitDataSet(dataSet1,dataSet2,test_proportion = 0.2):
    m = np.shape(dataSet1)[0]
    total = list(range(m))
    test = random.sample(range(m),int(test_proportion * m))
    for i in range(len(test)):
        total.remove(test[i])
    return dataSet1[test,:],dataSet2[test,:],dataSet1[total,:],dataSet2[total,:]
a = np.eye(10)
c, d, e, f = splitDataSet(a,a)