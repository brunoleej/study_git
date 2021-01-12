#!/usr/bin/env python
# coding: utf-8

# In[14]:


import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# In[3]:


iris = load_iris()


# In[5]:


iris_df = pd.DataFrame(data = iris.data,columns = iris.feature_names)
iris_df['label'] = iris.target
iris_df['label'].value_counts()


# In[7]:


kfold = KFold(n_splits = 3)
n_iter = 0
for train_index, test_index in kfold.split(iris_df):
    n_iter += 1
    label_train = iris_df['label'].iloc[train_index]
    label_test = iris_df['label'].iloc[test_index]
    print('## 교차 검증: {0}'.format(n_iter))
    print('학습 레이블 데이터 분포:\n',label_train.value_counts())
    print('검증 레이블 데이터 분포:\n',label_test.value_counts())


# In[10]:


# Straified KFold
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits = 3)
n_iter = 0

for train_index, test_index in skf.split(iris_df,iris_df['label']):
    n_iter += 1
    label_train = iris_df['label'].iloc[train_index]
    label_test = iris_df['label'].iloc[test_index]
    print('## 교차 검증: {0}'.format(n_iter))
    print('학습 레이블 데이터 분포:\n',label_train.value_counts())
    print('검증 레이블 데이터 분포:\n',label_test.value_counts())


# In[16]:


clf = DecisionTreeClassifier(random_state = 156)

skfold = StratifiedKFold(n_splits =3)
n_iter = 0
cv_accuracy = []

# StratifiedKFold의 split()호출 시 반드시 레이블 데이터 세트도 추가 입력 필요
for train_index,test_index in skf.split(iris_df,iris_df['label']):
    # split()으로 반환된 인덱스를 이용해 학습용, 검증용 테스트 데이터 추출
    x_train,x_test = iris.data[train_index],iris.data[test_index]
    y_train,y_test = iris.target[train_index],iris.target[test_index]
    # 학습 및 예측
    clf.fit(x_train,y_train)
    y_pred = clf.predict(x_test)
    
    # 반복 시마다 accuracy 측정
    accuracy = np.round(accuracy_score(y_test,y_pred),4)
    train_size = x_train.shape[0]
    test_size = x_test.shape[0]
    print('\n#{0} 교차 검증 정확도: {1}, 학습 데이터 크기: {2},검증 데이터 크기: {3}'.format(n_iter,accuracy,train_size,test_size))
    print('#{0} 검증 세트 인덱스: {1}'.format(n_iter,test_index))
    cv_accuracy.append(accuracy)
    # 교차 검증별 정확도 및 평균 정확도 계산
    print('\n## 교차 검증별 정확도:',np.round(cv_accuracy,4))
    print('## 평균 검증 정확도: ',np.mean(cv_accuracy))

