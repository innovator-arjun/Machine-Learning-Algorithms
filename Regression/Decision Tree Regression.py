#!/usr/bin/env python
# coding: utf-8

# In[17]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# In[18]:


data_pos=pd.read_csv('data/Position_Salaries.csv')


# In[19]:


data_pos


# In[20]:


X=data_pos.iloc[:,1:2].values


# In[21]:


y=data_pos.iloc[:,2].values


# In[22]:


y


# In[23]:


plt.scatter(X,y,color='red')


# In[24]:


#Hence we have to follow non-linear regression


# In[25]:


from sklearn.tree import DecisionTreeRegressor


# In[26]:


regressor=DecisionTreeRegressor(random_state=0)


# In[27]:


regressor.fit(X,y)


# In[28]:


y_pred=regressor.predict(6)


# In[29]:


y_pred


# In[ ]:





# In[30]:


plt.scatter(X,y,color='red')
plt.plot(X,regressor.predict(X),color='blue')
plt.title('Truth or bluf using Decision Tree Regressor')


# In[31]:


#Something is wrong,Red Flag. It is not continous curve

# hence we have to see in high resolution


# In[32]:


X_grid=np.arange(min(X),max(X),0.01)

X_grid=X_grid.reshape((len(X_grid),1))
plt.scatter(X,y,color='red')
plt.plot(X_grid,regressor.predict(X_grid),color='blue')
plt.title('Truth or bluf using Decision Tree Regressor')


# In[33]:


# if we predict 6.3 or 6.5 or 5.8 it will give the same result as 15,000. 
#Since it takes average in the split


# In[34]:


pred=regressor.predict(6.5)
pred


# In[35]:


pred=regressor.predict(6.3)
pred


# In[36]:


pred=regressor.predict(5.9)
pred


# In[ ]:




