#coding:UTF-8

"""
弱分类器
"""
import os,sys 
import numpy as np

#载入数据
def loadData():
    dataMat=[0,1,2,3,4,5,6,7,8,9]
    classLables=[1,1,1,-1,-1,-1,1,1,1,-1]
    return dataMat,classLables


#生成弱分类器,给定数据，根据x>thre或x<thre产生，其阈值使得该分类器
#在训练数据上的分类误差率最小
def weakClf(traindata,classlables,weight):  #参数为训练数据和权值分布
    weakC={}
    m=np.shape(traindata)[0]   #训练样本总数
    err1=[0,0,0,0,0,0,0,0,0,0]      #分类误差率
    err2=[0,0,0,0,0,0,0,0,0,0]
    classow1=np.zeros((m,10))   #学习得到的分类数据
    classow2=np.zeros((m,10))
  

    #假设弱分类器由x>thre或x<thre产生，阈值thre使得该分类器在训练数据集上分类误差率最低
    for i in range(m):  
        thre=traindata[i]
        #print "thre:",thre


        if (thre==0):
            for p in range(m):
                classow1[i][p]=-1
                classow2[i][p]=1
        else:
            for p in range(thre):
                classow1[i][p]=1
                classow2[i][p]=-1
            for p in range(m-thre):
                classow1[i][thre+p]=-1
                classow2[i][thre+p]=1

        for k in range(m):
            if (classow1[i][k]!=classlables[k]):
                err1[i]+=weight[k]
        for k in range(m):
            if (classow2[i][k]!=classlables[k]):
                err2[i]+=weight[k]		
            
    
    #确定误差率，并且找出误差率最小的阈值及弱分类器，计算此弱分类器的系数
    if (min(err1)<min(err2)):   
        err=min(err1)
        thre=err1.index(min(err1))
        way=1      #当way=1时，表示x<thre时分为1，x>thre时分为-1
        classow=classow1[thre]

    else:
        err=min(err2)
        thre=err2.index(min(err2))
        way=-1    #当way=-1时，表示x<thre时分为-1，x>thre时分为1
        classow=classow2[thre]
    
    #弱分类器
    weakC["thre"]=thre
    weakC["classresult"]=classow
   
    return weakC,err,classow


def adaboost(dataArr,classLables,t=20):
    weakClfArr=[]        #弱分类器数组
    n=np.shape(dataArr)[0]     #样本的数目
    #D=[1.0/n,1.0/n,1.0/n,1.0/n,1.0/n,1.0/n,1.0/n,1.0/n,1.0/n,1.0/n]    #创建权值分布矩阵，大小为n*1，值全为1/n
    D=np.ones(n)/n      #创建权值分布矩阵，大小为n*1，值全为1/n
    aggclass=np.zeros(n)

    print "original data:",dataArr
    print "original lable:",classLables,"\n"
    #对于i=1,2....t
    for i in range(t):

        #打印输出
        print '\n%sthe %dth result%s'%('*'*25,i+1,'*'*25)

        #弱分类器分类
        weakC,err,classow = weakClf(dataArr,classLables,D)
       
        print "threshould:",weakC["thre"]
        print "error rate:",err
        print "classify result:",classow.T
        
        #计算弱分类器权值
        cindex=0.5*(np.log((1-err)/max(err,1e-16)))
        weakC["index"]=cindex
        weakClfArr.append(weakC)
        print "index of this weak classifier :",cindex

        #更新权值分布
        print "current weight:",D
        expin=np.multiply(-1*cindex*np.array(classLables),np.array(classow))
        D=np.multiply(D,np.exp(expin))
        D=D/D.sum()
        print "new weight:",D

        #计算弱分类器加权累计值，f(x)=ɑ1*G1(x)+ɑ2*G2(x)....
        aggclass+=cindex*classow
        print "aggregrat value:",aggclass.T

        #计算误差
        aggclassErr=np.multiply(np.sign(aggclass)!=np.array(classLables),np.array(np.ones(n)))
        errClassNum=aggclassErr.sum()
        print "misclassified data:",aggclassErr
        print"error number:",errClassNum,"\n"

        if errClassNum == 0:
            break

    return weakClfArr


if __name__=='__main__':
    datMat,classLables=loadData()
    adaboost(datMat,classLables,6)



               

