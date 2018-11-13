# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 10:15:54 2018

@author: avaithil
"""

import pandas as pd
import numpy as np

dataset=pd.read_csv('titanic.csv')
dataset.head()

fig=dataset.Fare.hist(bins=10)

fig=dataset.boxplot(column='Fare',by='Survived')

dataset.Fare.describe()

IQR=dataset.Fare.quantile(0.75) -  dataset.Fare.quantile(0.25)

Lower_fence=dataset.Fare.quantile(0.25)-(IQR*1.5)

Upper_fance=dataset.Fare.quantile(0.75) + (IQR*1.5)

print(IQR)

print(dataset[dataset.Fare>100].shape[0])
print(dataset[dataset.Fare>200].shape[0])
print(dataset[dataset.Fare>300].shape[0])

dataset[dataset.Fare>300]

fig=dataset.Age.hist(bins=10)

fig=dataset.boxplot(column='Age',by='Survived')

dataset.Age.describe()


print(len(dataset.Sex.unique()))
print(len(dataset.Ticket.unique()))
print(len(dataset.Cabin.unique()))

cabin_dict={k:i for i,k in enumerate(dataset.Cabin.unique(),0)}

dataset['Cabin_mapped']=dataset['Cabin'].map(cabin_dict)

dataset[['Cabin_mapped','Cabin']].head()

dataset['Sex']=dataset['Sex'].map({'male':0 ,'female':1})