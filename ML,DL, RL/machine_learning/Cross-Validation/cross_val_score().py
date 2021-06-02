#!/usr/bin/env python
# coding: utf-8

# In[15]:


import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score,cross_validate

iris = load_iris()
clf = DecisionTreeClassifier(random_state = 156)

# 성능 지표는 정확도(accuracy), 교차 검증 세트는 3개
scores = cross_val_score(clf,iris.data,iris.target,scoring = 'accuracy',cv = 3)
print('교차 검증별 정확도:', np.round(scores,4))
print('평균 검증 정확도:',np.round(np.mean(scores),4))

