#!/usr/bin/env python
# coding: utf-8

# In[62]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns


# In[63]:


boston = pd.read_csv('./Boston_house.csv')
boston


# In[64]:


boston_data = boston.drop(['Target'],axis = 1)
boston_data


# In[65]:


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


# # crim, rm,lstat을 통한 다중 선형 회귀분석

# In[66]:


x_data = boston[['CRIM','RM','LSTAT']]
target = boston[['Target']]
x_data.head()


# In[67]:


target.head()


# In[68]:


# 상수항 추가
x_data1 = sm.add_constant(x_data,has_constant='add')


# In[69]:


multi_model = sm.OLS(target,x_data1)
fitted_multi_model = multi_model.fit()


# In[70]:


fitted_multi_model.summary()


# # crim, rm, lstat, b, tax, age, zn, nox, indus 변수를 통한 다중선형회귀분석

# In[71]:


# boston data에서 원하는 변수만 뽑아오기
x_data2 = boston[['CRIM','RM',"LSTAT","B",'TAX','AGE','ZN','NOX','INDUS']]
x_data2.head()


# In[72]:


# 상수항 추가
x_data2_ = sm.add_constant(x_data2,has_constant='add')


# In[73]:


# 회귀모델 적합
multi_model2 = sm.OLS(target,x_data2_)
fitted_multi_model2 = multi_model2.fit()


# In[74]:


# 결과 출력
fitted_multi_model2.summary()
# nox, indus의 p-value값이 너무 높음


# In[75]:


# 세 변수만 추가한 모델의 회귀 계수
fitted_multi_model.params


# In[76]:


# full 모델의 회귀 계수
fitted_multi_model2.params


# In[77]:


# base 모델과 full 모델의 잔차비교
fitted_multi_model.resid.plot(label = 'full')
fitted_multi_model2.resid.plot(label = 'full_add')
plt.legend()
plt.show()


# # 상관계수/산점도를 통해 다중공선성 확인

# In[78]:


# 상관행렬
x_data2.corr()


# In[79]:


# 상관행렬 시각화해서 보기
cmap = sns.light_palette('darkgray',as_cmap=True)
sns.heatmap(x_data2.corr(),annot = True,cmap = cmap)
plt.show()
# nox,indus의 correlation값이 굉장히 강함


# In[80]:


# 변수별 산점도 시각화
sns.pairplot(x_data2)
plt.show()


# # VIF를 통한 다중공선성 확인

# In[81]:


from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
vif['VIF Factor'] = [variance_inflation_factor(x_data2.values,i) for i in range(x_data2.shape[1])]
vif['features'] = x_data2.columns
vif


# In[82]:


# nox변수 제거 후(x_data3) VIF확인

vif = pd.DataFrame()
x_data3 = x_data2.drop(['NOX'],axis = 1)
vif['VIF Factor'] = [variance_inflation_factor(x_data3.values,i)for i in range(x_data3.shape[1])]
vif['features'] = x_data3.columns
vif
# vif가 10이 넘으면 보통 다중공선성이 있다고 봄


# In[83]:


# RM변수 제거 후(x_data4) VIF확인
vif = pd.DataFrame()
x_data4 = x_data3.drop(['RM'],axis = 1)
vif['VIF Factor'] = [variance_inflation_factor(x_data4.values,i)for i in range(x_data4.shape[1])]
vif['features'] = x_data4.columns
vif


# In[84]:


# nox변수 제거한 데이터(x_data3) 상수항 추가 후 회귀 모델 적합
# nox, rm 변수 제거한 데이터(x_data4) 상수항 추가 후 회귀 모델 적합
x_data3_ = sm.add_constant(x_data3,has_constant='add')
multi_model3 = sm.OLS(target,x_data3_)
fitted_multi_model3 = multi_model3.fit()

x_data4_ = sm.add_constant(x_data4,has_constant='add')
multi_model4 = sm.OLS(target,x_data4_)
fitted_multi_model4 = multi_model4.fit()


# In[85]:


fitted_multi_model3.summary()


# In[86]:


fitted_multi_model4.summary()


# # 학습 데이터와 검증 데이터 분할

# In[116]:


from sklearn.model_selection import train_test_split

x = x_data2_
y = target

x_train,x_test,y_train,y_test = train_test_split(x,y,train_size = 0.7,test_size = 0.3, random_state = 1)
print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)


# In[117]:


# train_x 회귀모델 적합
x_train.head()
fit_1 = sm.OLS(y_train,x_train)
fit_1 = fit_1.fit()


# In[118]:


# 검증 데이터에 대한 예측값과 true값 비교
plt.plot(np.array(fit_1.predict(x_test)),label = 'pred')
plt.plot(np.array(y_test),label = 'true')
plt.legend()
plt.show()


# In[119]:


# x_data3와 x_data4 학습 검증데이터 분할
x2 = x_data3_
y = target

x2_train,x2_test,y2_train,y2_test = train_test_split(x2,y,train_size = 0.7,test_size = 0.3,random_state = 1)


# In[120]:


fit_2 = sm.OLS(y2_train,x2_train)
fit_2 = fit_2.fit()


# In[121]:


plt.plot(np.array(fit_2.predict(x2_test)),label = 'pred')
plt.plot(np.array(y2_test),label = 'true')
plt.legend()
plt.show()


# In[122]:


x3 = x_data4_
y = target

x3_train,x3_test,y3_train,y3_test = train_test_split(x3,y,train_size = 0.7, test_size = 0.3,random_state = 1)


# In[123]:


fit_3 = sm.OLS(y3_train,x3_train)
fit_3 = fit_3.fit()


# In[124]:


plt.plot(np.array(fit_3.predict(x3_test)),label = 'pred')
plt.plot(np.array(y3_test),label = 'true')
plt.legend()
plt.show()


# In[125]:


# true값과 예측값 비교
plt.plot(np.array(fit_2.predict(x2_test)),label = 'pred')
plt.plot(np.array(fit_3.predict(x3_test)),label = 'pred')
plt.plot(np.array(y_test),label = 'true')
plt.legend()
plt.show()


# In[128]:


# full모델 시각화 해서 비교
plt.plot(np.array(fit_1.predict(x_test)),label = 'pred')
plt.plot(np.array(fit_2.predict(x2_test)),label = 'pred_vif')
plt.plot(np.array(fit_3.predict(x3_test)),label = 'pred_vif2')
plt.plot(np.array(y_test),label = 'true')
plt.legend()
plt.show()


# In[129]:


# 잔차 계산
plt.plot(np.array(y2_test['Target'] - fit_1.predict(x_test)),label = 'pred_full')
plt.plot(np.array(y2_test['Target'] - fit_2.predict(x2_test)),label = 'pred_vif')
plt.plot(np.array(y2_test['Target'] - fit_3.predict(x3_test)),label = 'pred_vif2')
plt.legend()
plt.show()


# # MSE를 통한 검증테이터에 대한 성능비교

# In[130]:


from sklearn.metrics import mean_squared_error


# In[131]:


mean_squared_error(y_true= y_test['Target'],y_pred = fit_1.predict(x_test))


# In[137]:


# 변수 1개 제거
mean_squared_error(y_true= y_test['Target'],y_pred = fit_2.predict(x2_test))


# In[136]:


# 변수 2개 제거
mean_squared_error(y_true= y_test['Target'],y_pred = fit_3.predict(x3_test))

