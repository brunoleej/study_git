#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import statsmodels.api as sm
# Statsmodels는 사용자가 데이터를 탐색하고 통계 모델을 추정하며 통계 테스트를 수행 할 수있는 Python 패키지입니다.


# In[3]:


boston = pd.read_csv('./Boston_house.csv')


# In[4]:


boston.head()


# In[5]:


# target 제외한 데이터만 뽑기
boston_data = boston.drop(['Target'],axis = 1)


# In[6]:


boston_data.describe()


# In[7]:


'''
타겟 데이터
1978 보스턴 주택 가격
506개 타운의 주택 가격 중앙값 (단위 1,000 달러)

특징 데이터
CRIM: 범죄율
INDUS: 비소매상업지역 면적 비율
NOX: 일산화질소 농도
RM: 주택당 방 수
LSTAT: 인구 중 하위 계층 비율
B: 인구 중 흑인 비율
PTRATIO: 학생/교사 비율
ZN: 25,000 평방피트를 초과 거주지역 비율
CHAS: 찰스강의 경계에 위치한 경우는 1, 아니면 0
AGE: 1940년 이전에 건축된 주택의 비율
RAD: 방사형 고속도로까지의 거리
DIS: 직업센터의 거리
TAX: 재산세율'''


# # crim / rm / lstat 세개의 변수로 각각 단순 선형 회귀 분석하기

# In[49]:


# 변수 설정 target/crim/rm/lstat
target = boston[['Target']]
crim = boston[['CRIM']]
rm = boston[['RM']]
lstat = boston[['LSTAT']]


# # target ~ crim 선형회귀분석

# In[50]:


#crim 변수에 상수항 추가하기
crim1 =sm.add_constant(crim, has_constant = 'add')
crim1


# In[51]:


# sm.OLS 적합시키기
model1 = sm.OLS(target, crim1)
fitted_model1 = model1.fit()


# In[52]:


# summary함수를 통해서 결과 출력
fitted_model1.summary()


# In[53]:


# 회귀 계수 출력
fitted_model1.params 


# # y_hat = beta0 + beta1 * X 계산해보기

# In[54]:


# 회귀 계수 x 데이터(X)
np.dot(crim1,fitted_model1.params)


# In[55]:


# predict함수를 통해 yhat구하기
pred1 = fitted_model1.predict(crim1)


# In[56]:


# 직접구한 yhat과 predict함수를 통해 구한 yhat차이
np.dot(crim1, fitted_model1.params) - pred1


# # 적합시킨 직선 시각화

# In[61]:


import matplotlib.pyplot as plt
plt.yticks(fontname = 'Arial')
plt.scatter(crim, target, label = 'data')
plt.plot(crim, pred1, label = 'result')
plt.legend()
plt.show()


# In[62]:


plt.scatter(target,pred1)
plt.xlabel('real_value')
plt.ylabel('pred_value')
plt.show()


# In[63]:


# residual 시각화
fitted_model1.resid.plot()
plt.xlabel('residual_number')
plt.show()


# In[64]:


# 잔차의 합계산해보기
np.sum(fitted_model1.resid)


# # 위와 동일하게 rm변수와 lstat변수로 각각 단순 선형회귀분석 적합시켜보기

# In[71]:


# 상수항추가
rm1 = sm.add_constant(rm,has_constant = 'add')
lstat1 = sm.add_constant(lstat,has_constant = 'add')


# In[76]:


# 회귀모델 적합
model2 = sm.OLS(target,rm1)
fitted_model2 = model2.fit()

model3 = sm.OLS(target,lstat1)
fitted_model3 = model3.fit()


# In[77]:


# rm모델 결과 출력
fitted_model2.summary()


# In[78]:


# lstat모델 결과 출력
fitted_model3.summary()


# In[80]:


# 각각의 yhat_ 예측하기
pred2 = fitted_model2.predict(rm1)
pred3 = fitted_model3.predict(lstat1)


# In[81]:


# rm 모델 시각화
plt.scatter(rm,target,label = 'data')
plt.plot(rm,pred2,label = 'result')
plt.legend()
plt.show()


# In[82]:


# lstat 모델 시각화
plt.scatter(lstat,target,label = 'data')
plt.plot(lstat,pred3,label = 'result')
plt.legend()
plt.show()


# In[83]:


# rm모델 residual 시각화
fitted_model2.resid.plot()
plt.xlabel('residual_number')
plt.show()


# In[84]:


# lstat 모델 residual 시각화
fitted_model3.resid.plot()
plt.xlabel('residual_number')
plt.show()


# In[87]:


# 세가지 모델의 residual 비교
fitted_model1.resid.plot(label = 'crim')
fitted_model2.resid.plot(label = 'rm')
fitted_model3.resid.plot(label = 'lstat')
plt.legend()
plt.show()


# In[ ]:




