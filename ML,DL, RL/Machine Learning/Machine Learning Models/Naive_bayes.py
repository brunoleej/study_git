#!/usr/bin/env python
# coding: utf-8

# # Gaussian Naive Bayes

# In[2]:


import pandas as pd
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB


# In[5]:


iris = load_iris()
x = pd.DataFrame(iris.data)
y = pd.DataFrame(iris.target)


# In[7]:


x.shape,y.shape


# In[8]:


x.head()


# In[9]:


y.head()


# In[36]:


# model Object Generate
model = GaussianNB()
# model fitting
fitted = model.fit(iris.data,iris.target)
# model predict
y_pred = fitted.predict(iris.data)


# In[37]:


# fitted.predict_proba check
# 3-class problem
fitted.predict_proba(iris.data)[[1,32,54,100]]


# In[38]:


fitted.predict(iris.data)[[1,32,54,100]]


# #  Confusion_matrix

# In[39]:


from sklearn.metrics import confusion_matrix


# In[41]:


confusion_matrix(iris.target,y_pred)


# # Prior 설정하기

# In[60]:


# model Object Gnerate
model2 = GaussianNB(priors=[1/100,1/100,98/100])
# model fitting
fitted2 = model2.fit(iris.data,iris.target)
# model predict
y2_pred = model2.predict(iris.data)


# In[62]:


fitted2.predict_proba(iris.data)[[1,32,54,100]]


# In[63]:


fitted2.predict(iris.data)[[1,32,54,100]]


# In[64]:


confusion_matrix(iris.target,y2_pred)


# In[66]:


model3 = GaussianNB(priors = [1/100,98/100,1/100])
fitted3 = model3.fit(iris.data,iris.target)
y3_pred = fitted3.predict(iris.data)
confusion_matrix(iris.target,y3_pred)


# # Multinomial Naive Bayes

# In[67]:


from sklearn.naive_bayes import MultinomialNB


# In[68]:


import numpy as np


# In[69]:


x = np.random.randint(5,size=(6,100))
y = np.array([1,2,3,4,5,6])


# In[70]:


x


# In[71]:


y


# In[76]:


# model Object Generate
multi_model = MultinomialNB()
# model fitting
multi_model.fit(x,y)


# In[80]:


# model predict
multi_model.predict(x[2:3])


# In[81]:


# model.predict_proba check
multi_model.predict_proba(x[2:3])


# In[87]:


multi_model2 = MultinomialNB(class_prior=[0.1,0.5,0.1,0.1,0.1,0.1])
multi_model2.fit(x,y)


# In[88]:


multi_model2.predict_proba(x[2:3])

