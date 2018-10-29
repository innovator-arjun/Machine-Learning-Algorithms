#!/usr/bin/env python
# coding: utf-8

# In[111]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# In[112]:


data_startup=pd.read_csv('data/50_Startups.csv')


# In[113]:


data_startup.head()


# In[114]:


data_startup.isnull().sum()


# In[115]:


data_startup[['R&D Spend','Administration','Marketing Spend','Profit']]=data_startup[['R&D Spend','Administration','Marketing Spend','Profit']].astype(int)


# In[116]:


data_startup.head()


# In[117]:


#[Matrix]#Independent Variable 'R&D Spend','Administration','Marketing Spend',State
#{Vector}#Dependent Variable : Profit


# In[118]:


X=data_startup.iloc[:,:-1].values
y=data_startup.iloc[:,4].values


# In[119]:


from sklearn.preprocessing import LabelEncoder,OneHotEncoder


# In[120]:


labelencoder_X=LabelEncoder()
X[:,3]=labelencoder_X.fit_transform(X[:,3])
onehotencoder=OneHotEncoder(categorical_features=[3])


# In[121]:


X=onehotencoder.fit_transform(X).toarray()


# In[122]:


X=X[:,1:]


# In[123]:


from sklearn.cross_validation import train_test_split


# In[124]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)


# In[125]:


from sklearn.linear_model import LinearRegression


# In[126]:


regressor=LinearRegression()


# In[127]:


regressor.fit(X_train,y_train)


# In[128]:


#Predicting the test set results


# In[129]:


y_pred=regressor.predict(X_test)


# In[130]:


diff=y_pred.astype(int)-y_test


# In[ ]:





# In[131]:


#Building the  optimal model using backward Elimination


# In[132]:


import statsmodels.formula.api as sm


# In[133]:


X=np.append(arr=np.ones((50,1)).astype(int),values=X,axis=1)


# In[137]:


X.astype(int)


# In[ ]:




