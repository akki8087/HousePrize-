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
        del co[i]
        
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


null2 = X_test.isnull().sum()
collnull2 = []
for i in range(len(null2)):
    if null2[i] > 0:
        collnull2.append(null2.index[i])
        
li2 = list(collnull2)        

li2.remove('GarageYrBlt')
for d in li2:
    if type(X_test[d][2]) == str :
        X_test[d] = X_test[d].fillna(list(X_test[d].mode())[0])
    else:
        X_test[d] = X_test[d].fillna(X_test[d].mean())

X_test['GarageYrBlt'] = X_test.GarageYrBlt.fillna(list(X_test['GarageYrBlt'].mode())[0])

both = X.append(X_test)

both_dummy = pd.get_dummies(both,drop_first = True)

yr = ['YearBuilt','YearRemodAdd','GarageYrBlt','YrSold']
for r in yr:
    both_dummy[r] = 2010 - both_dummy[r]
    
X_d = both_dummy.iloc[:1460,:]
X_t = both_dummy.iloc[1460:,:]

from sklearn.cross_validation import train_test_split
X_train, X_testt, y_train, y_test = train_test_split(X_d, y, test_size = 1/3, random_state = 0)

'''
from sklearn.linear_model import  LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

'''
# RandomForest 
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor()
regressor.fit(X_train,y_train)

y_pred = regressor.predict(X_testt)

'''
#Xgboost
import xgboost as xgb
from xgboost.sklearn import XGBRegressor
regressor = XGBRegressor(learning_rate =0.01, n_estimators=15)
regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_testt)

'''

import math
rmse = math.sqrt(np.mean((np.log(y_pred) - np.log(y_test))**2))

y_result = regressor.predict(X_t)



result = pd.DataFrame()

result['Id'] = test['Id']

result['SalePrice'] = y_result

result.to_csv('HP1.csv',index = False)
