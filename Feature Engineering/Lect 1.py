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



























