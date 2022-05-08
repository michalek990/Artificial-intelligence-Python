# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pandas as pd



#punkt 1
print("Punkt 1")
data_csv = pd.read_csv("practice_lab_1.csv",sep=";")
tablica = data_csv.values
tablica_1 = tablica[1::2]
tablica_2 = tablica[::2]
roznica = tablica_1 - tablica_2
print(roznica)

#punkt 2 
print("Punkt 2")
srednia = tablica.mean()
odchylenie = tablica.std()
tablica_minus_srednia = tablica - srednia 
tablica_minus_odchylenie = tablica - odchylenie
print(tablica_minus_srednia)
print(tablica_minus_odchylenie)

#punkt 3 
print("Punkt 3")
srednie_kolumn = np.spacing(tablica.mean(axis=0))
odchylenie_kolumn = np.spacing(tablica.std(axis=0))

#punkt 4
print("Punkt 4")
wspolczynnik_zmiennosci = np.spacing(srednie_kolumn/odchylenie_kolumn)

#punkt 5
print("Punkt 5")
m = wspolczynnik_zmiennosci.max()
arg = wspolczynnik_zmiennosci.argmax()

#punkt 6
print("Punkt 6")

num_rows, num_cols = tablica.shape
res = np.zeros(num_cols)
for i in range(0,num_cols):
    tab_col = tablica[:,i]
    maska = tab_col==0
    tab_masked = tab_col[maska]
    l = tab_masked.size
    res[i]=l
print(res)

#punkt 7 
print("Punkt 7")
m = res.max()
for i in range(0,num_cols):
    if(res[i]==m):
        print(data_csv.columns[i])
        
        
#Punkt 8
print("Punkt 8")
for i in range(0, num_cols):
    tab_col = tablica[:,i]
    maska = tab_col==0
    tab_masked = tab_col[maska]
    l = tab_masked.size
    res[i]=l  
print(res)
m = res.max()
for i in range (0,num_cols):
    if(res[i]==m):
        print(data_csv.columns[i])
        
  
#punkt 9 
print("Punkt 9")
for i in range (0,num_cols):
    tab_col = tablica[:,i]
    parzyste = tablica[::2]
    nieparzyste = tablica[1::2]
    if parzyste.sum()> nieparzyste.sum():
        res[i]=1
    else:
        res[i]=0
    
print(res)
m = res.max()
for i in range(0 ,num_cols):
    if(res[i]==m):
         print(data_csv.columns[i])
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    