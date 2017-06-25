#coding:UTF-8

"""
载入简单的数据
"""

import numpy as np 

def loadData():
	dataMat=np.matrix([0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0])
	classLables=[1,1,1,-1,-1,-1,1,1,1-1]
    return dataMat,classLables