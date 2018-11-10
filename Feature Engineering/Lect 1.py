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