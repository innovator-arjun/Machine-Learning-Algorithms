#!/usr/bin/env python
# coding: utf-8

# In[73]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[74]:


data_mall=pd.read_csv('Mall_Customers.csv')


# In[75]:


data_mall.head()


# In[76]:


#Spending Score-->1 (Customer uses less money in spending)
#Spending Score-->100 (Customer uses more money in spending)


# In[77]:


X=data_mall.iloc[:,2:4].values


# In[78]:


#Using the elbow method to find the optimal number of clusters


# In[79]:


from sklearn.cluster import KMeans


# In[80]:


wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')


# In[81]:


#Applying k-means to the mall dataset


# In[82]:


kmeans=KMeans(n_clusters=5,init='k-means++',max_iter=300,n_init=10,random_state=0)


# In[83]:


y_means=kmeans.fit_predict(X)


# In[84]:


y_means


# In[85]:


#Visualising the  clusters


# In[86]:


plt.scatter(X[y_means==0,0], X[y_means==0,1],s=100,c='red',label='Careless')
plt.scatter(X[y_means==1,0], X[y_means==1,1],s=100,c='blue',label='Standard')
plt.scatter(X[y_means==2,0], X[y_means==2,1],s=100,c='green',label='Target')
plt.scatter(X[y_means==3,0], X[y_means==3,1],s=100,c='cyan',label='Carefull')
plt.scatter(X[y_means==4,0], X[y_means==4,1],s=100,c='magenta',label='Sensible')

plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c='yellow',label='Cluster 1')


plt.title('Cluster of clients')

plt.xlabel('Annual Income')

plt.ylabel('Spending Score')

plt.legend()
plt.show()


# In[ ]:




