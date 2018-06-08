# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 12:26:01 2018

@author: NP
"""

import pandas as pd
import numpy as np

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

X = train.iloc[:,1:-1]
y = train.iloc[:,-1]

X_test = test.iloc[:,1:]
null = X.isnull().sum()

combo = [X, X_test]
col = ['PoolQC','Fence','MiscFeature','Alley']
for co in combo:
    for i in col:
        del combo[co][i]
null1 = X.isnull().sum()
collnull = []
for i in range(len(null1)):
    if null1[i] > 0:
        collnull.append(null1.index[i])
        
li = list(collnull)        
#li = ['MasVnrType','BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
 #'Electrical', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']
li.remove('GarageYrBlt')
for d in li:
    if type(X[d][2]) == str :
        X[d] = X[d].fillna(list(X[d].mode())[0])
    else:
        X[d] = X[d].fillna(X[d].mean())

X['GarageYrBlt'] = X.GarageYrBlt.fillna(list(X['GarageYrBlt'].mode())[0])

X_dummy = pd.get_dummies(X,drop_first = True)

yr = ['YearBuilt','YearRemodAdd','GarageYrBlt','YrSold']
for r in yr:
    X_dummy[r] = 2010 - X_dummy[r]
    

from sklearn.linier_model import  LinearRegressor
regressor = LinearRegressor()
regressor.fit(X_dummy,y)






