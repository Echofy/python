#coding:UTF-8

"""
Adaboost算法
"""

import math
import numpy as np

def adaboost(dataArr,classLables,t=20):
    weakClfArr=[]        #弱分类器数组
    n=np.shape(dataArr)[0]     #样本的数目
    D=np.mat(np.ones((n,1)))/n    #创建权值分布矩阵，大小为n*1，值全为1/n
    aggclass=np.mat(np.zeros((n,1))) #
    #对于i=1,2....t
    for i in range(t):
        #弱分类器分类
        weakC,err,classow = weakClf(dataArr,classlables,D)
        print "Data:",D
        print "class of weakClf:",classow.T
        
        #计算弱分类器权值
        cindex=0.5*(np.log((1-err)/max(err,1e-16)))
        weakC["index"]=cindex
        weakClfArr.append(weakC)

        #更新权值分布
        expin=np.multiply(-1*cindex*np.mat(classLables).T,classow)
        D=np.multiply(D,np.exp(expin))
        D=D/D.sum()

        #计算弱分类器加权累计值，f(x)=ɑ1*G1(x)+ɑ2*G2(x)....
        aggclass+=cindex*classow
        print "aggregrat value:",aggclass.T

        #计算误差
        aggclassErr=np.multiply(np.sign(aggclass)!=np.mat(classLables).T,np.ones((n,1)))
        errClassNum=aggclassErr.sum()
        print"error number:",errClassNum,"\n"

        if errClassNum == 0:
            break

    return weakClfArr









  








