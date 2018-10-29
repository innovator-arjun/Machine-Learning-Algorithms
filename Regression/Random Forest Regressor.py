#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# In[5]:


data_pos=pd.read_csv('data/Position_Salaries.csv')


# In[6]:


data_pos


# In[15]:


X=data_pos.iloc[:,1:2].values


# In[16]:


y=data_pos.iloc[:,2].values


# In[17]:


y


# In[18]:


X


# In[19]:


from sklearn.ensemble import RandomForestRegressor


# In[65]:


regressor=RandomForestRegressor(n_estimators=300,random_state=0)


# In[66]:


regressor.fit(X,y)


# In[67]:


y_pred=regressor.predict(6.5)


# In[68]:


y_pred


# In[69]:


plt.scatter(X,y,color='red')
plt.plot(X,regressor.predict(X),color='blue')

plt.title('Random Forest Prediction')

plt.xlabel('Position Level')
plt.ylabel('Salary ')

plt.show()


# In[70]:


#Since it is a non-continous model

X_grid=np.arange(min(X),max(X),0.1)

X_grid=X_grid.reshape(len(X_grid),1)
plt.scatter(X,y,color='red')
plt.plot(X_grid,regressor.predict(X_grid),color='blue')

plt.title('Random Forest Prediction')

plt.xlabel('Position Level')
plt.ylabel('Salary ')

plt.show()


# In[72]:


#if tree increases the steps won't increase, since it will take average from all the 
#tree prediction.


#Random Forest is a collection of many Decision Tree.
#We have ,
#1. linear model
#2. non-linear model
#3. non-linear ,non-continous model
#4. non-linear, non-continous, ensemble model


# In[ ]:




