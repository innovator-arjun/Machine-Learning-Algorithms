# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 10:23:42 2018

@author: avaithil
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset=pd.read_csv('Churn_Modelling.csv')


dataset.Gender.value_counts()

dataset.Geography.value_counts()

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

#Feature Scaling to avoid domination of independent variable over antoher

from sklearn.preprocessing import StandardScaler

sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)


#Part-2 Let's make the ANN!
import keras

from keras.models import Sequential
from keras.layers import Dense

#Sequential is ANN class-Initialising the ANN
classifier=Sequential()

#11 input layer --> 11 independent column

#we will use rectifier funtion as an activation hidden layer
#Sigmoid to find probability at the output layer
#Dense will assign a random number close to 0 not 0

#Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim=6,kernel_initializer='uniform',activation='relu',input_dim=11))

(11+1)/2#output_dim
#take the avg of input and output to select the node for Hidden layer

#Adding the second hidden layer
classifier.add(Dense(output_dim=4,kernel_initializer='uniform',activation='relu'))
(1+11)/2


#Adding the output layer
#change the activation to sigmoid to find the probability
classifier.add(Dense(output_dim=1,kernel_initializer='uniform',activation='sigmoid'))

#Compiling the ANN
#Optimizer--> Gradient  stochastic gradient
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#Fitting the ANN to the training set
classifier.fit(X_train,y_train,nb_epoch=150,batch_size=15)


#Part 3 Making the prediction and evaluation the model


y_pred=classifier.predict(X_test)
#Y_pred is in the for of probability, so we should use threshold.
y_pred=(y_pred>0.5)

from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test,y_pred)


(1534+189)/2000


















