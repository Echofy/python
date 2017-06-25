#coding:UTF-8

"""
Boosting算法
"""

import os,sys 
import numpy as np
import random

#载入数据
def loadData():
    dataMat=[0,1,2,3,4,5,6,7,8,9]
    classLables=[1,1,1,-1,-1,-1,1,1,1,-1]
    return dataMat,classLables

#基分类器训练算法
def weakClf(traindata,classlables):  #参数为训练数据
    weakC={}
    m=np.shape(traindata)[0]     #训练样本总数
    err1=[0 for i in range(m)]   #错分数目，
    err2=[0 for i in range(m)]
    classow1=[[0 for i in range(m)] for j in range(m)]   #学习得到的分类数据
    classow2=[[0 for i in range(m)] for j in range(m)]

    #假设弱分类器由x>thre或x<thre产生，阈值thre使得该分类器在训练数据集上分类误差率最低
    for i in range(m):  
 
        if (i==0):
            for p in range(m):
                classow1[i][p]=-1
                classow2[i][p]=1
        else:
            for p in range(i):
                classow1[i][p]=1
                classow2[i][p]=-1
            for p in range(m-i):
                classow1[i][i+p]=-1
                classow2[i][i+p]=1

        for k in range(m):
            if (classow1[i][k]!=classlables[k]):
                err1[i]+=1
        for k in range(m):
            if (classow2[i][k]!=classlables[k]):
                err2[i]+=1        
            
    
    #确定误差率，并且找出误差率最小的阈值及弱分类器，计算此弱分类器的系数
    if (min(err1)<min(err2)):   
        err=min(err1)
        thre=err1.index(min(err1))
        classow=classow1[thre]

    else:
        err=min(err2)
        thre=err2.index(min(err2))
        classow=classow2[thre]
    

    #弱分类器,thre:阈值V  classresult：弱分类器分类结果
    weakC["thre"]=thre   
    weakC["classresult"]=classow
   
    print "threshould:",thre
    print "misclassified number:",err   
    print "result of weakclassfier:",classow
   
    return weakC,err,classow

#弱分类器（weakclassifier）对某一数据（data）分类所得的结果（标签）
def ClassifyByWeak(data,weakclassfier):
    if(data<weakclassfier["thre"]):
        classlable=weakclassfier["classresult"][0]
    else:
        classlable=weakclassfier["classresult"][-1]
    return classlable
 
#Boosting算法
#参数为（训练数据，训练数据标签，第一个数据集的样本容量，第二个数据集的样本容量）
def Boosting(dataArr,classLables,n1,n2):
    weakClfArr=[]             #弱分类器集合
    n=np.shape(dataArr)[0]    #原始样本总量
    dataArr3=[]
    classlables3=[]
    dataArr4=[]
    classlables4=[]
    budataArr=[]
    buclasslables=[]
    
    #输出原始训练数据
    print "the training data(D):",dataArr
    print "lables of training data:",classLables,"\n"

    #抽取第一个数据集D1，从原始样本中不放回地随机选取n1个样本点
    dataArrNum1=random.sample(range(n),n1)  
    dataArr1=[0 for i in range(n1)]  
    classlables1=[0 for i in range(n1)]
    for (i,j) in zip(range(n1),dataArrNum1): 
        dataArr1[i]=dataArr[j]
        classlables1[i]=classLables[j]
    
    #输出第一个数据集D1
    print "the first dataset(D1):",dataArr1
    print "lables of the first dataset(D1):",classlables1 
    
    #根据第一个数据集训练第一个弱分类器
    weakC1,err1,classow1 = weakClf(dataArr1,classlables1) 
    weakClfArr.append(weakC1)  
    print "the first weak classifier:",weakC1,"\n"  
    
    #抽取n1个样本后的剩余样本
    dataArrNum2=list(set(range(n))-set(dataArrNum1))  
    dataArr2=[0 for i in range(n-n1)] 
    classlables2=[0 for i in range(n-n1)] 
    for (i,j) in zip(range(n-n1),dataArrNum2):
        dataArr2[i]=dataArr[j]
        classlables2[i]=classLables[j]

    #抽取第二个数据集D2
    while len(dataArr3)<n2:   #保证生成n2个样本的数据集D2   
        r=random.random()   #生成随机数，作为判定投掷硬币的正反面

        #当硬币为正面时，如果是正面就选取D 中剩余样本点一个一个送到C中进行分
        #类，遇到第一个被错分的样本加入集合D2中
        if r<0.5:
            #print "1-len of dataArr2:",len(dataArr2),"\n"
            #判定当剩余数据集不空时，利用第一个弱分类器对剩余样本进行分类，
            if len(dataArr2)>0:   
                for i in range(len(dataArr2)) :  #对于剩余样本中每个样本
                    classiedlable=ClassifyByWeak(dataArr2[i],weakC1)

                    #若是错分样本则加入到数据集D2中，并且将其从剩余样本数据集中删除
                    #跳出，掷下一次色子，删除是因为不放回
                    if classiedlable!=classlables2[i]:
                        dataArr3.append(dataArr2[i])
                        classlables3.append(classlables2[i])
                        dataArr2.remove(dataArr2[i])
                        classlables2.remove(classlables2[i])
                        break

                    #若是分对的样本，则将其加入到备份数据集中，并且将其从剩余样本数据集中删除
                    #跳出，掷下一次色子，删除是因为不放回    
                    else:
                        budataArr.append(dataArr2[i])
                        buclasslables.append(classlables2[i])
                        dataArr2.remove(dataArr2[i])
                        classlables2.remove(classlables2[i])
                        break  

        #当硬币为反面时，就选取一个被C1正确分类的样本点加入集合D2中
        else:
            #print "2-len of dataArr2:",len(dataArr2),"\n"  

            #判断当剩余数据集不空时，找到一个分对的样本加入D2中
            
            if len(dataArr2)>0: 
                for i in range(len(dataArr2)):

                    #若备份数据不空，则在备份数据中拿，加入数据集D2，并将其从剩余样本中删除
                    #跳出，掷下一次色子，删除是因为不放回    

                    if len(budataArr)!=0:
                        dataArr3.append(budataArr[0])
                        classlables3.append(buclasslables[0])
                        budataArr.remove(budataArr[0])
                        buclasslables.remove(buclasslables[0])
                        break

                    #否则，用弱分类器找出分对的样本，加入数据集D2，并将其从剩余样本中删除   
                    #跳出，掷下一次色子，删除是因为不放回    s
                    elif ClassifyByWeak(dataArr2[i],weakC1)==classlables2[i]:
                        dataArr3.append(dataArr2[i])
                        classlables3.append(classlables2[i])
                        dataArr2.remove(dataArr2[i])
                        classlables2.remove(classlables2[i])
                        break

            #当剩余数据集中空时，只能在备份数据集中找分对的样本            
            else:
                dataArr3.append(budataArr[0])
                classlables3.append(buclasslables[0])
                budataArr.remove(budataArr[0])
                buclasslables.remove(buclasslables[0])

    #print "budataArr:",budataArr
    #print "buclasslables:",buclasslables,"\n"

    #print "dataArr2(1):",dataArr2
    #print "classlables2(1):",classlables2,"\n"
    
    #输出第二个数据集D2
    print "the second dataset(D2):",dataArr3
    print "lables of the second dataset(D2):",classlables3

    #利用第二个数据集训练第二个弱分类器
    weakC2,err2,classow2=weakClf(dataArr3,classlables3)
    weakClfArr.append(weakC2)
    print "the second weak classifier:",weakC2,"\n"
    
    #剩余样本
    for i in range(len(budataArr)):
        dataArr2.append(budataArr[i])
        classlables2.append(buclasslables[i])

    #print "dataArr2(2):",dataArr2
    #print "classlables2(2):",classlables2,"\n"

    #抽取第三个数据集D3
    #如C1和C2分类结果不同，就把该样本加入集合D3
    for i in range(len(dataArr2)):
        classiedlable1=ClassifyByWeak(dataArr2[i],weakC1)
        classiedlable2=ClassifyByWeak(dataArr2[i],weakC2)
        if classiedlable2!=classiedlable1:
            dataArr4.append(dataArr2[i])
            classlables4.append(classlables2[i])
    
    #若C1和C2恰好可以正确分类剩余样本，则无数据集D3，也就没有第三个弱分类器
    if len(dataArr4)==0:
        print "there is no third dataset!"
        print "there is no third weak classifier!","\n"

    ##若C1和C2都不能正确分类剩余样本，则将这类样本加入数据集D3
    else:
        print "the third dataset(D3):",dataArr4
        print "lables of the third dataset(D4):",classlables4

        #用第三个数据集D3训练第三个弱分类器C3
        weakC3,err3,classow3=weakClf(dataArr4,classlables4)
        weakClfArr.append(weakC3)
        print "the third weak classifier:",weakC3,"\n"

    print "the final classifier:","\n",weakClfArr,"\n"    

    return weakClfArr

#用Boosting算法进行分类
def ClassifyByBoosting(data,weakClfArr):
    if ClassifyByWeak(data,weakClfArr[0])==ClassifyByWeak(data,weakClfArr[1]):
        classlableB=ClassifyByWeak(data,weakClfArr[0])
    else:
        classlableB=ClassifyByWeak(data,weakClfArr[2])
    return classlableB


if __name__=='__main__':
    datMat,classLables=loadData()
    weakClfArr=Boosting(datMat,classLables,4,3)
    classlable=ClassifyByBoosting(8,weakClfArr)
    print "Boosting classlable of 8 is:",classlable


        

               


 
