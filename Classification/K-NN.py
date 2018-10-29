#!/usr/bin/env python
# coding: utf-8

# In[93]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[94]:


data_social=pd.read_csv('Social_Network_Ads.csv')


# In[95]:


X=data_social.iloc[:,2:4].values


# In[96]:


X


# In[97]:


y=data_social.iloc[:,4].values


# In[98]:


y


# In[99]:


from sklearn.cross_validation import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25)


# In[ ]:





# In[100]:


from sklearn.preprocessing import StandardScaler


# In[101]:


sc=StandardScaler()


# In[102]:


X_train=sc.fit_transform(X_train)


# In[103]:


X_test=sc.fit_transform(X_test)


# In[104]:


from sklearn.neighbors import KNeighborsClassifier


# In[105]:


classifier=KNeighborsClassifier(n_neighbors=5,p=2,metric='minkowski')


# In[106]:


classifier.fit(X_train,y_train)


# In[107]:


y_pred=classifier.predict(X_test)


# In[108]:


y_pred


# In[109]:


y_test


# In[110]:


from sklearn.metrics import confusion_matrix


# In[111]:


cm=confusion_matrix(y_test,y_pred)


# In[112]:


cm


# In[113]:


#Visualizing the  training set


# In[114]:


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
plt.title('K-NN (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


# In[116]:


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
plt.title('K-NN (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


# In[ ]:




