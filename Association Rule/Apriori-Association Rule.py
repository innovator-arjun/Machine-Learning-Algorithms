#!/usr/bin/env python
# coding: utf-8

# In[74]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[75]:


data_market=pd.read_csv('Market_Basket_Optimisation.csv',header=None)
#since we dont have any column name, python maight consider row values as header. So we are using header=None


# In[76]:


data_market.head()


# In[77]:


#the dataset contains 7500 transaction from the super market


# In[78]:


# We have to import the data set in a different way for Apriori. It wont work in dataframe
#Apirori--> list


# In[79]:


transactions=[]
# it will create a empty list or vector
# 0 to 7500
for i in range(0,7501):
    transactions.append([str(data_market.values[i,j]) for j in range(0,20)])
    #to add each and every row in the list or vector


# In[80]:



transactions[0:2]


# In[81]:


from apyori import apriori


# In[82]:


#It will take transactions as input and give rules as output
#min_support , (3 time as day*7(week))/7500(total transaction)=0.0028
rules=apriori(transactions,min_support=0.003,min_confidence=0.2,min_lift=3,min_length=2 )





# In[83]:


#visualising the results
#min_support-->Transactions
#confidence--> probability of buying the item
# min_lift--> compared with random crowd, what is the percent increase.


# In[84]:


results=list(rules)


# In[85]:


results


# In[ ]:




