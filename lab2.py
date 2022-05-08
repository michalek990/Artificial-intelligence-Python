# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error

#zadanie1
data=pd.read_csv('housing.csv', sep=';')
kolumny = list(data.columns)
arr=data.values
corrArr=data.corr()

def plotXY3(x,y):
    fig,ax=plt.subplots(1,1,figsize=(10,10))
    ax.scatter(x,y)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    
for i in range(0,arr.shape[1]-1):
    x=arr[:,i]
    y=arr[:,-1]
    plotXY3(x,y)

#zadanie 2

def testRegresion(data,howMany):
    res = np.zeros(howMany)
    X=data[:,:-1]
    y=data[:,-1]
    for i in range(1,howMany+1):
        X_train, X_test , y_train, y_test = train_test_split(X,y,test_size=0.2, shuffle=False)
        linReg=LinearRegression()
        linReg.fit(X_train,y_train)
        y_pred=linReg.predict(X_test)
        mase=mean_absolute_percentage_error(y_test,y_pred)
        res[i-1]=mase
    m=res.mean()
    return m
    
data=pd.read_csv('housing.csv', sep=';')
kolumny=list(data.columns)
data=data.values
howMany=20
res=testRegresion(data,howMany)
print(res)


#zadanie 3

def delOutliers(y_train):
    outliers = np.abs((y_train-y_train.mean())/y_train.std())>2.5
    y_train_mean=y_train.copy()
    y_train_mean[outliers]=y_train.mean()
    return y_train_mean

def testRegresion(data,howMany):
    res = np.zeros(howMany)
    X=data[:,:-1]
    y=data[:,-1]
    for i in range(1,howMany+1):
        X_train, X_test , y_train, y_test = train_test_split(X,y,test_size=0.2, shuffle=False)
        y_train = delOutliers(y_train)
        linReg=LinearRegression()
        linReg.fit(X_train,y_train)
        y_pred=linReg.predict(X_test)
        mase=mean_absolute_percentage_error(y_test,y_pred)
        res[i-1]=mase
    m=res.mean()
    return m

#zadanie 4
nowe_cechy = np.stack(data[:.7]*data[:,1],axis=-1)
data_additional=np.concatenate([np.expadn_dims(nowe_cechy,axis=-1),data],axis=-1)
res2 =testRegresion(data_additional,howMany)
print(res2)

#zadanie 5
from sklearn.datasets import load_diabetes
dat=load_diabetes()
df = pd.DataFrame(dat.data, columns = dat.feature_names)
dataVal=df.values()
dataCol= list(df.columns)
X= np.copy(dataVal)
y=dat.target

data= np.concatenate([X,np.expand_dims(y,axis=-1)],axis=-1)
howMany=20
res4=testRegresion(data,howMany)
print(res4)