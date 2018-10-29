#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# In[5]:


data_ads=pd.read_csv('Social_Network_Ads.csv')


# In[6]:


data_ads.head(20)


# In[7]:


X=data_ads.iloc[:,[2,3]].values
X


# In[8]:


y=data_ads.iloc[: , 4].values


# In[9]:


y


# In[10]:


data_ads.shape


# In[11]:


#100 for test set
#300 for training set


# In[12]:


from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)


# In[13]:


#Feature Scalling for accurate result


# In[14]:


from sklearn.preprocessing import StandardScaler


# In[15]:


sc_X=StandardScaler()

X_train=sc_X.fit_transform(X_train)

X_test=sc_X.fit_transform(X_test)


# In[16]:


from sklearn.linear_model import LogisticRegression


# In[17]:


classifier=LogisticRegression(random_state=0)


# In[18]:


classifier.fit(X_train,y_train)


# In[19]:


#Predicting the test set result


# In[21]:


y_pred=classifier.predict(X_test)


# In[22]:


y_pred


# In[23]:


X_test


# In[24]:


#since it is scaled we cannot directly interpret the result


# In[25]:


from sklearn.metrics import confusion_matrix


# In[26]:


cm=confusion_matrix(y_test,y_pred)


# In[33]:


cm

#11 incorrect prediction


# In[28]:


#Visualising the training set results


# In[31]:


# red and green points are truth
#region are prediction by the system
#line is called as prediction boundary and it is not random
#linear classification is done. The line is the best by the system.


# In[32]:


#logistic regression is a linear classifier


# In[36]:


# Visualising the Training set results


# In[35]:


from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


# In[37]:


# Visualising the Test set results


# In[38]:


from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


# In[ ]:




