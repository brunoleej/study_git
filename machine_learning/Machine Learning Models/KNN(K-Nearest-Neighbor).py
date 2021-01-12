#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix


# In[8]:


iris = load_iris()
x = iris.data[:,:2]
y = iris.target


# In[9]:


# model Object Generate
model = KNeighborsClassifier(5)
# model fitting
model.fit(x,y)


# In[10]:


# model predict
y_pred = model.predict(x)


# In[11]:


confusion_matrix(y,y_pred)


# # Cross-Validation을 활용한 최적의 k찾기

# In[12]:


# module import 
from sklearn.model_selection import cross_val_score


# In[15]:


k_range = range(1,100)
k_scores = []

for k in k_range:
    knn = KNeighborsClassifier(k)
    scores = cross_val_score(knn,x,y,cv=10,scoring='accuracy')
    k_scores.append(scores.mean())


# In[16]:


plt.plot(k_range,k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-validated accuracy')
plt.show()


# # Weight를 준 kNN

# In[18]:


from sklearn.neighbors import KNeighborsRegressor


# In[19]:


np.random.seed(0)
x = np.sort(5 * np.random.rand(40,1),axis = 0)
T = np.linspace(0,5,500)[:,np.newaxis]
y = np.sin(x).ravel()
y[::5] += 1 * (0.5 - np.random.rand(8))


# In[24]:


knn = KNeighborsRegressor(n_neighbors)
y_ = knn.fit(x,y).predict(T)


# In[25]:


n_neighbors = 5

for i, weights in enumerate(['uniform','distance']):
    knn = KNeighborsRegressor(n_neighbors,weights = weights)
    y_ = knn.fit(x,y).predict(T)
    
    plt.subplot(2,1, i + 1)
    plt.scatter(x,y,c='k',label = 'data')
    plt.plot(T,y_,c='g',label = 'prediction')
    plt.axis('tight')
    plt.legend()
    plt.title("KNeighborsRegressor (k = %i, weights = '%s')" %(n_neighbors,weights))
    
plt.tight_layout()
plt.show()

