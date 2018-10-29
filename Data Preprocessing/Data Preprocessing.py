#!/usr/bin/env python
# coding: utf-8

# In[55]:


import pandas as pd

#To take care of missing data
from sklearn.preprocessing import Imputer

from sklearn.model_selection import train_test_split


# In[22]:


data=pd.read_csv('data/Data.csv')


# In[23]:


#To print the dataset in jupyter notebook
data


# In[24]:


X=data.iloc[:,:-1].values


# In[25]:


X


# Independent Variable--> Country,Age , Salary
# 
# Dependent Variable--> Purchased
# 
# We will use  Independent Variable to Predict Dependent Variable in Machine Learning.

# In[26]:


imputer=Imputer(missing_values='NaN',strategy='mean',axis=0)


# In[27]:


imputer=imputer.fit(X[:,1:3])


# In[28]:


X[:,1:3]=imputer.transform(X[:,1:3])


# In[29]:


age_average=data['Age'].mean()
salary_average=data['Salary'].mean()


# In[30]:


data['Age']=data['Age'].fillna(age_average)
data['Salary']=data['Salary'].fillna(salary_average)


# In[31]:


data


# Country and Purchased are categorical Variable
# 
# Since Machine Learning model are mathematical, we cannot fit the text .
# 
# So we will convert the text into numeric.
# http://queirozf.com/entries/one-hot-encoding-a-feature-on-a-pandas-dataframe-an-example

# In[33]:


df1=pd.get_dummies(data['Country'])


# In[34]:


df1


# System will think Spain(0) is lesser than germany(2) , this might mislead the system.
# 
# Hence we should use Dummy Variable to prevent it.

# In[35]:


frames=[data,df1]
frames


# In[36]:


result=pd.concat(frames,axis=1)


# In[37]:


result


# In[43]:


result=result.drop('Country',axis=1)


# In[44]:


result


# In[45]:


df2=pd.get_dummies(data['Purchased'])
df2


# In[46]:


frame2=[result,df2]


# In[47]:


frame2


# In[48]:


result=pd.concat(frame2,axis=1)


# In[ ]:





# In[53]:


result_test=result[['No','Yes']]


# In[70]:


result=result.drop('No',axis=1)
result=result.drop('Yes',axis=1)
result=result.drop('Purchased',axis=1)


# In[71]:


X_train,X_test,y_train,y_test=train_test_split(result,result_test,test_size=0.2)


# In[72]:


X_train

#It has independent vairable for train


# In[73]:


X_test

#It has independent vairable for test


# In[74]:


y_train

#It hold dependent variable for training


# In[75]:


y_test

#It hold Dependent vairable for testing


# In[ ]:





# In[ ]:





# In[ ]:


#Creating dummy variable in python for categorical data


# In[56]:


df=pd.DataFrame({'A':['a','b','c'],'B':['b','a','c']})


# In[57]:


df


# In[58]:


one_hot=pd.get_dummies(df['B'])


# In[59]:


one_hot


# In[60]:


df=df.drop('B',axis=1)


# In[61]:


df


# In[62]:


df=df.join(one_hot)


# In[63]:


df


# In[ ]:




