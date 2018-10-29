#!/usr/bin/env python
# coding: utf-8

# In[46]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# In[47]:


data_pos=pd.read_csv('data/Position_Salaries.csv')


# In[48]:


data_pos


# In[49]:


data_pos.isnull().sum()


# In[50]:


sns.barplot(x='Level',y='Position',data=data_pos)


# In[51]:


sns.barplot(x='Level',y='Salary',data=data_pos)


# In[52]:


sns.barplot(x='Salary',y='Position',data=data_pos)


# In[ ]:





# In[53]:


data_pos.shape


# In[54]:


data_pos.describe


# In[55]:


data_pos.head()


# In[56]:


X=data_pos.iloc[:,1:2].values

#:2 is aded to consider X as an Matrix


# In[57]:


y=data_pos.iloc[:,2].values


# In[58]:


X


# In[59]:


y


# In[60]:


plt.scatter(X,y,color='red')


# In[61]:


#hence the graph result is curve, we cannot use the normal approach
#since we have less number of dataset, no need to split the test and train data.
#We have to go for a non-linear model-->polynomial Regression


# In[62]:


# no feature scaling, most of the non-linear model itself will do the feature scaling


# In[ ]:





# In[63]:


#Fitting Linear Regression to the dataset


# In[64]:


from sklearn.linear_model import LinearRegression


# In[65]:


linear_reg=LinearRegression()


# In[66]:


linear_reg.fit(X,y)


# In[ ]:





# In[85]:


#Building polynomial regression to the dataset


# In[68]:


from sklearn.preprocessing import PolynomialFeatures


# In[69]:


poly_reg=PolynomialFeatures(degree=5)


#Degree 3 imporves the accuracy than Degree 2, Degree 4 or Degree 5 is more accurate than Degree 3


# In[70]:


X_poly=poly_reg.fit_transform(X)


# In[71]:


X_poly


# In[72]:


X


# In[73]:


lin_reg_2=LinearRegression()


# In[74]:


lin_reg_2.fit(X_poly,y)


# In[ ]:





# In[75]:


#Visualising  the Linear Regression results


# In[76]:


plt.scatter(X,y,color='red')
plt.plot(X,linear_reg.predict(X),color='blue')


plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.title('Truth of Bluff (Linear Regression)')


# In[ ]:





# In[ ]:





# In[77]:


#Visualising the Polynomial regression results


# In[78]:



plt.scatter(X,y,color='red')
plt.plot(X,lin_reg_2.predict(X_poly),color='blue')

# we have to use X_poly when we use lin_reg_2, since we used transform


plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.title('Truth of Bluff (Polynomial Regression)')


# In[79]:


data_pos


# In[80]:


#Predicting a new result with Linear Regression


# In[81]:


linear_reg.predict(6)

# Not a good prediction


# In[ ]:





# In[82]:


#Predicting a new result with polynomial regression


# In[83]:


lin_reg_2.predict(poly_reg.fit_transform(6))

#Very close prediction of Region Manger.


# In[ ]:





# In[84]:


###The difference is so high and it wont work if we follow the same steps in Simple or Multiple regression.


# In[ ]:




