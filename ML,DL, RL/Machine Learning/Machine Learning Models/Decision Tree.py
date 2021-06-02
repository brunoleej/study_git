#!/usr/bin/env python
# coding: utf-8

# # 의사결정나무(Decision tree)

# In[1]:


from sklearn.tree import DecisionTreeClassifier


# In[2]:


x = [[0,0],[1,1]]
y = [0,1]


# In[3]:


# model Object Generate
model = DecisionTreeClassifier()
model.fit(x,y)


# In[4]:


model.predict([[1,1]])


# In[15]:


from sklearn.datasets import load_iris
from sklearn import tree
import graphviz


# In[9]:


iris = load_iris()


# In[11]:


iris.data.shape,iris.target.shape


# # Decision Tree 구축 및 시각화

# In[25]:


# model Object Generate
model = DecisionTreeClassifier()
# model fitting
model.fit(iris.data,iris.target)# predict
y_pred = model.predict(iris.data)


# In[19]:


# tree visualization
dot_data = tree.export_graphviz(model,out_file = None,
                                feature_names = iris.feature_names,
                               class_names = iris.target_names
                                ,filled = True, rounded = True,
                               special_characters=True)
graph = graphviz.Source(dot_data)
graph


# - 엔트로피를 활용한 트리

# In[26]:


# model Object Generate
model2 = DecisionTreeClassifier(criterion='entropy')
# model fitting
model2.fit(iris.data,iris.target)
# predict
y2_pred = model2.predict(iris.data)


# In[21]:


dot_data2 = tree.export_graphviz(model2,out_file = None,
                                feature_names=iris.feature_names,
                                class_names = iris.target_names,
                                filled = True,rounded= True,
                                special_characters=True)
graph2 = graphviz.Source(dot_data2)
graph2


# - 프루닝

# In[33]:


# model Object Generate
model3 = DecisionTreeClassifier(criterion='entropy',max_depth=2)
# model fitting
model3.fit(iris.data,iris.target)
# predict
y3_pred = model3.predict(iris.data)


# In[34]:


dot_data3 = tree.export_graphviz(model3,out_file = None,
                                feature_names = iris.feature_names,
                                class_names = iris.target_names,
                                filled = True, rounded = True,
                                special_characters='True')
graph3 = graphviz.Source(dot_data3)
graph3


# # confusion matrix

# In[35]:


from sklearn.metrics import confusion_matrix


# In[36]:


confusion_matrix(iris.target,y_pred)


# In[37]:


confusion_matrix(iris.target,y2_pred)


# In[38]:


confusion_matrix(iris.target,y3_pred)


# # Train,Test 구분 및 Confusion matrix 계산

# In[41]:


from sklearn.model_selection import train_test_split


# In[53]:


x_train,x_test,y_train,y_test = train_test_split(iris.data,iris.target,
                                                 stratify = iris.target,
                                                 random_state = 1)


# In[54]:


model4 = DecisionTreeClassifier(criterion='entropy')
model4.fit(x_train,y_train)
y4_pred = model4.predict(x_test)


# In[55]:


confusion_matrix(y_test,y4_pred)


# # Decision regression tree

# In[40]:


import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt


# In[60]:


rng = np.random.RandomState(1)
x = np.sort(5 * rng.rand(80, 1), axis=0)
y = np.sin(x).ravel()
y[::5] += 3 * (0.5 - rng.rand(16))


# - regression tree 구축

# In[61]:


regr1=tree.DecisionTreeRegressor(max_depth=2)
regr2=tree.DecisionTreeRegressor(max_depth=5)


# In[62]:


regr1.fit(x,y)


# In[63]:


regr2.fit(x,y)


# In[64]:


X_test=np.arange(0.0,5.0,0.01)[:,np.newaxis]
X_test


# In[65]:


y_1=regr1.predict(X_test)
y_2=regr2.predict(X_test)


# In[66]:


plt.figure()
plt.scatter(X, y, s=20, edgecolor="black",
            c="darkorange", label="data")
plt.plot(X_test, y_1, color="cornflowerblue",
         label="max_depth=2", linewidth=2)
plt.plot(X_test, y_2, color="yellowgreen", label="max_depth=5", linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Decision Tree Regression")
plt.legend()
plt.show()


# In[67]:


dot_data4 = tree.export_graphviz(regr2, out_file=None, 
                                filled=True, rounded=True,  
                                special_characters=True)
graph4 = graphviz.Source(dot_data4) 
graph4


# In[68]:


dot_data5 = tree.export_graphviz(regr1, out_file=None, 
                                filled=True, rounded=True,  
                                special_characters=True)
graph5=graphviz.Source(dot_data5)
graph5


# In[ ]:




