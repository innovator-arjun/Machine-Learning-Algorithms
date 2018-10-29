#!/usr/bin/env python
# coding: utf-8

# In[63]:


#SVM is a linear classsifier

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[64]:


data_soc=pd.read_csv('Social_Network_Ads.csv')


# In[65]:


data_soc.head()


# In[66]:


#data_soc.Gender[data_soc.Gender=='Male']=0


# In[67]:


#data_soc.Gender[data_soc.Gender=='Female']=1


# In[68]:


#data_soc


# In[69]:


X=data_soc.iloc[:,2:4].values
X


# In[70]:


y=data_soc.iloc[:,4].values


# In[71]:


from sklearn.cross_validation import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0,test_size=0.25)


# In[72]:


from sklearn.preprocessing import StandardScaler


# In[73]:


sc=StandardScaler()


# In[74]:


X_train=sc.fit_transform(X_train)

X_test=sc.fit_transform(X_test)


# In[75]:


from sklearn.svm import SVC


# In[76]:


classifier=SVC(kernel='linear',random_state=0,degree=5)

#kernal can be linear or rbf to make SVM linear or non-linear


# In[77]:


classifier.fit(X_train,y_train)


# In[78]:


y_pred=classifier.predict(X_test)


# In[79]:


y_pred


# In[80]:


y_test


# In[81]:


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)


# In[82]:


cm


# In[83]:


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
plt.title('SVM (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


# In[84]:


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
plt.title('SVMn (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


# In[ ]:




