# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 01:08:08 2018

@author: ARJUN_PC
"""

import pandas as pd
import numpy as np

use_col=['loan_amnt','int_rate','annual_inc','open_acc','loan_status','open_il_12m']

dataset=pd.read_csv('loan.csv',usecols=use_col).sample(10000,random_state=44)

dataset.head()

#Continuous variable
fig=dataset.loan_amnt.hist(bins=50)

fig=dataset.int_rate.hist(bins=30)


#discrete variable

fig=dataset.open_acc.hist(bins=100)

dataset.open_il_12m.unique()

fig=dataset.open_il_12m.hist(bins=50)

#creating a new column(dependent variable)

dataset['defaulter']=np.where(dataset.loan_status.isin(['Default']),1,0)

dataset.defaulter.unique()


use_cols=['id','home_ownership','loan_status','purpose']
dataset=pd.read_csv('loan.csv',usecols=use_cols).sample(10000,random_state=44)

#categorical variable
dataset.home_ownership.unique()

dataset.home_ownership.value_counts().plot.bar()

dataset.purpose.value_counts().plot.bar()

dataset.loan_status.value_counts().plot.bar()

use_cols=['loan_amnt','grade','purpose','issue_d','last_pymnt_d']

dataset=pd.read_csv('loan.csv',usecols=use_cols).sample(10000,random_state=44)

dataset.head()

dataset.dtypes
#convetring object to date for date time columns

dataset['issue_dt']=pd.to_datetime(dataset.issue_d)


dataset['last_pymnt_dt']=pd.to_datetime(dataset.last_pymnt_d)


dataset.dtypes

fig=dataset.groupby(['issue_dt','grade'])['loan_amnt'].sum().unstack().plot(figsize=(14,8))



titanic_dataset=pd.read_csv('titanic.csv')


titanic_dataset.isnull().sum()

titanic_dataset['age_null']=np.where(titanic_dataset.Age.isnull(),1,0)

titanic_dataset.groupby(['Survived'])['age_null'].mean()
titanic_dataset.groupby(['Sex'])['age_null'].mean()


titanic_dataset[titanic_dataset['Embarked'].isnull()]


titanic_dataset.Cabin.isnull().sum()

titanic_dataset['cabin_null']=np.where(titanic_dataset.Cabin.isnull(),1,0)

titanic_dataset.groupby(['Survived'])['cabin_null'].mean()





dataset=pd.read_csv('loan.csv',usecols=['emp_title','emp_length'])
dataset.head()

dataset.emp_title.unique()[1:20]


dataset.emp_length.value_counts()/len(dataset)*100

dataset.emp_length.unique()


length_dict={k:'0-10 years' for k in dataset.emp_length.unique()}
length_dict['10+ years']='10+ years'
length_dict['n/a']='n/a'
dataset['emp_length_redefined']=dataset.emp_length.map(length_dict)


value=len(dataset[dataset.emp_title.isnull()])

dataset[dataset.emp_title.isnull()].groupby(['emp_length_redefined'])['emp_length'].count().sort_values()/value









































