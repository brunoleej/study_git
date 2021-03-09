#!/usr/bin/env python
# coding: utf-8

# In[42]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm


# In[43]:


# 데이터 불러오기
boston = pd.read_csv('./Boston_house.csv')
boston


# In[44]:


# Target column Extraction
boston_data = boston.drop(['Target'],axis = 1)
boston_data


# In[45]:


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


# In[46]:


target = boston[['Target']]
crim = boston[['CRIM']]
rm = boston[['RM']]
lstat = boston[['LSTAT']]


# In[47]:


crim1 = sm.add_constant(crim,has_constant='add')


# In[48]:


model1 = sm.OLS(target,crim1)
fitted_model1 = model1.fit()


# In[49]:


fitted_model1.summary()


# In[50]:


pred1 = fitted_model1.predict(crim1)


# In[51]:


plt.scatter(crim,target,label = 'data')
plt.plot(crim,pred1,label='result')
plt.legend()
plt.show()


# In[52]:


plt.scatter(target,pred1)
plt.xlabel('real_value')
plt.ylabel('pred_value')
plt.show()


# In[53]:


fitted_model1.resid.plot()
plt.xlabel('residual_number')
plt.show()


# In[54]:


rm1 = sm.add_constant(rm,has_constant='add')
lstat1 = sm.add_constant(lstat,has_constant='add')


# In[55]:


model2 = sm.OLS(target,rm1)
fitted_model2 = model2.fit()
model3 = sm.OLS(target,lstat1)
fitted_model3= model3.fit()


# In[56]:


fitted_model2.summary()


# In[57]:


fitted_model3.summary()


# In[58]:


pred2 = fitted_model2.predict(rm1)
pred3 = fitted_model3.predict(lstat1)


# In[59]:


plt.scatter(rm,target,label='data')
plt.plot(rm,pred2,label='result')
plt.legend()
plt.show()


# In[61]:


plt.scatter(lstat,target,label = 'data')
plt.plot(lstat,pred3, label = 'result')
plt.legend()
plt.show()


# In[62]:


fitted_model2.resid.plot()
plt.xlabel('residual_number')
plt.show()


# In[63]:


fitted_model2.resid.plot()
plt.xlabel('residual_number')
plt.show()


# In[66]:


fitted_model1.resid.plot(label = 'crim')
fitted_model2.resid.plot(label = 'rm')
fitted_model2.resid.plot(label ='lstat')
plt.legend()
plt.show()


# # 다중 선형회귀분석

# In[72]:


# boston data에서 crim, rm, lstat 변수만 뽑아오기
x_data = boston[['CRIM','RM','LSTAT']]
x_data.head()


# In[74]:


# 상수항 추가
x_data1 = sm.add_constant(x_data,has_constant='add')


# In[75]:


# 회귀모델 적합
multi_model = sm.OLS(target,x_data1)
fitted_multi_model = multi_model.fit()


# In[76]:


# summary함수를 통해 결과출력
fitted_multi_model.summary()


# # 단순선형회귀모델의 회귀계수와 비교

# In[77]:


# 단순선형회귀모델의 회귀 계수
print(fitted_model1.params)
print(fitted_model2.params)
print(fitted_model3.params)


# In[78]:


# 다중선형회귀의 회귀 계수
fitted_multi_model.params


# # 행렬연산을 통해 beta 구하기

# In[86]:


from numpy import linalg     # (X'X) - 1X'y


# In[87]:


ba = linalg.inv(np.dot(x_data1.T,x_data1))
np.dot(np.dot(ba,x_data1.T),target)


# In[88]:


# y_hat 구하기
pred4 = fitted_multi_model.predict(x_data1)


# In[91]:


# residual plot
fitted_model1.resid.plot(label = 'crim')
fitted_model2.resid.plot(label = 'rm')
fitted_model3.resid.plot(label = 'lstat')
fitted_multi_model.resid.plot(label = 'full')
plt.xlabel('residual_number')
plt.legend()
plt.show()

