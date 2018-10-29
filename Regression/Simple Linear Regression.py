#!/usr/bin/env python
# coding: utf-8

# In[18]:


import pandas as pd
from sklearn.cross_validation import train_test_split

import matplotlib.pyplot as plt


# In[2]:


data=pd.read_csv('data/Salary_Data.csv')


# In[3]:


data.head()
#YearsExperience-->independent Variable
#Salary-->dependent Variable


# In[4]:


data.shape


# In[5]:


data.isnull().sum()


# In[6]:


X=data.iloc[:,:-1].values


# In[7]:


y=data.iloc[:,1].values


# In[8]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/3,random_state=0)


# In[9]:


y_train


# In[10]:


from sklearn.linear_model import LinearRegression


# In[11]:


regressor=LinearRegression()


# In[12]:


regressor.fit(X_train,y_train)


# In[13]:


y_pred=regressor.predict(X_test)


# In[14]:


y_pred


# In[15]:


y_test


# In[17]:


#Some salary are over predicted
#We are goin gto plot some graphfor visualization.


# In[22]:


plt.scatter(X_train,y_train,color='red')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')


# In[30]:


plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')


# In[31]:


#Red are real salary 

#Blue line are Predicted Salary

#There is linear dependency betwwen experience and salary



# In[35]:


plt.scatter(X_test,y_test,color='red')
plt.plot(X_test,regressor.predict(X_test),color='blue')
plt.title('Salary vs Experience (Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')


# In[33]:


#New obeservation.
#Red are real salary
#Blue are predicted and it is accurate.


# In[ ]:




