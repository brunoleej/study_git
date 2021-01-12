#!/usr/bin/env python
# coding: utf-8

# In[2]:


# 모듈 임포트
import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold


# In[3]:


# iris 데이터 불러오기
iris = load_iris()


# In[5]:


# 모델 객체 생성
clf = DecisionTreeClassifier(random_state = 156)


# In[6]:


# 5개의 폴드 세트로 분리하는 KFold 객체와 플드 세트별 정확도를 담을 리스트 객체 생성
kfold = KFold(n_splits = 5)
cv_accuracy = []


# In[9]:


print('붓꽃 데이터 세트 크기: ',iris.data.shape[0])


# In[15]:


n_iter = 0

# KFold 객체의 split()을 호출하면 폴드 별 학습용, 검증용 테스트의 로우 인덱스를 array로 반환
for train_index, test_index in kfold.split(iris.data):
    # kfold.split()으로 반환된 인덱스를 이용해 학습용, 검증용 테스트 데이터 추출
    x_train,x_test = iris.data[train_index],iris.data[test_index]
    y_train,y_test = iris.target[train_index],iris.target[test_index]
    # 학습 및 예측
    clf.fit(x_train,y_train)
    y_pred = clf.predict(x_test)
    n_iter += 1
    # 반복 시마다 정확도 측정
    accuracy = accuracy_score(y_test,y_pred)
    train_size = x_train.shape[0]
    test_size = x_test.shape[0]
    print('/n#{0} 교차 검증 정확도: {1}, 학습 데이터 크기: {2}, 검증 데이터 크기: {3}'.format(n_iter,accuracy,train_size,test_size))
    print('#{0} 검증 세트 인덱스: {1}'.format(n_iter,test_index))      
    cv_accuracy.append(accuracy)

# 개별 iteration별 정확도를 합하여 평균 정확도 계산
print('\n## 평균 검증 정확도: ',np.mean(cv_accuracy))

