#!/usr/bin/env python
# coding: utf-8

# In[77]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[78]:


data_mall=pd.read_csv('Mall_Customers.csv')


# In[79]:


data_mall.head()


# In[80]:


X=data_mall.iloc[:,2:4].values


# In[81]:


X[0:5]


# In[82]:


#to find the optimal number of clusters-Dendrogram


# In[83]:


import scipy.cluster.hierarchy as sch


# In[84]:


dendro=sch.dendrogram(sch.linkage(X,method='ward'))

plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Eclidean Distance')
plt.show()


# In[85]:


#Fitting the hierarchical clustering to the mall dataset


# In[86]:


from sklearn.cluster import AgglomerativeClustering


# In[87]:


hc=AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage='ward')


# In[88]:


y_hc=hc.fit_predict(X)


# In[90]:


#Visualising the clusters


# In[92]:


plt.scatter(X[y_hc==0,0], X[y_hc==0,1],s=100,c='red',label='Careless')
plt.scatter(X[y_hc==1,0], X[y_hc==1,1],s=100,c='blue',label='Carefull')
plt.scatter(X[y_hc==2,0], X[y_hc==2,1],s=100,c='green',label='Target')
plt.scatter(X[y_hc==3,0], X[y_hc==3,1],s=100,c='cyan',label='Standard')
plt.scatter(X[y_hc==4,0], X[y_hc==4,1],s=100,c='magenta',label='Sensible')



plt.title('Cluster of clients')

plt.xlabel('Annual Income')

plt.ylabel('Spending Score')

plt.legend()
plt.show()


# In[ ]:




