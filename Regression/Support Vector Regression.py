#!/usr/bin/env python
# coding: utf-8

# In[155]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# In[156]:


data_pos=pd.read_csv('data/Position_Salaries.csv')


# In[157]:


X=data_pos.iloc[:,1:2].values
y=data_pos.iloc[:,2:3].values


# In[158]:


#We need to apply feature Scaling in RBF-Gaussian SVR


# In[159]:


from sklearn.svm import SVR


# In[166]:


regressor=SVR(kernel='poly',degree=4)

#kernal--> we have to choose the kernel like 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'
#rbf means gaussian


# In[167]:


regressor.fit(X,y)


# In[168]:


y_pred=regressor.predict(6.5)

#we need to tranform since we have used standard scaling


# In[169]:


y_pred


# In[170]:


plt.scatter(X,y,color='red')
plt.plot(X,regressor.predict(X),color='blue')

plt.title('Truth or bluff Support vector regressor')
plt.xlabel('Position level')

plt.ylabel('Salary')


# In[165]:


#CEO salary is considered as an outlier since it is far away


# In[ ]:




