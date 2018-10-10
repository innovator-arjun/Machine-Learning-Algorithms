# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 10:23:42 2018

@author: avaithil
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset=pd.read_csv('Churn_Modelling.csv')

X=dataset.iloc[:,3:13].values
y=dataset.iloc[:,13].values

#Part -1 -Data processing---> Categroical data -->gender,country

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
#Country 
labelencoder_X_1=LabelEncoder()
X[:,1]=labelencoder_X_1.fit_transform(X[:,1])

#Gender
labelencoder_X_2=LabelEncoder()
X[:,2]=labelencoder_X_2.fit_transform(X[:,2])

#We have to beware of dummy variable trap if we have 3 or more category

onehotencoder=OneHotEncoder(categorical_features=[1])
X=onehotencoder.fit_transform(X).toarray()


X=X[:,1:]

#Split the dataset

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)