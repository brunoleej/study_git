#!/usr/bin/env python
# coding: utf-8

# # SVM(Support Vector Machie)

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn import svm


# In[4]:


iris = load_iris()
x = iris.data[:,:2]
y = iris.target
C = 1


# In[5]:


model = svm.SVC(kernel = 'linear',C=C)
model.fit(x,y)


# In[7]:


y_pred = model.predict(x)


# In[6]:


from sklearn.metrics import confusion_matrix


# In[8]:


confusion_matrix(y,y_pred)


# # kernel SVM 적합 및 비교

# - LinearSVC

# In[11]:


model2 = svm.LinearSVC(C=C,max_iter=10000)
model2.fit(x,y)
y2_pred = model2.predict(x)
confusion_matrix(y,y2_pred)


# - radial basis function

# In[12]:


model3 = svm.SVC(kernel='rbf',gamma=0.7,max_iter=10000)
model3.fit(x,y)
y3_pred = model3.predict(x)
confusion_matrix(y,y3_pred)


# - polynomial kernel

# In[15]:


model4 = svm.SVC(kernel='poly',degree=3,C=C,gamma='auto')
model4.fit(x,y)
y4_pred = model4.predict(x)
confusion_matrix(y,y4_pred)

