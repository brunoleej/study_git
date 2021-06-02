# RobustScaler : 중위값을 기준으로 값을 전처리한다.
# 중위값을 기준으로 계산을 하기 때문에 중위값에서 멀리 떨어져 있는 이상치들의 영향력이 약해진다.
# 장점 : 이상치 제거를 하지 않은 상태에서 RobustScaler를 하면 이상치 제거의 효과가 있다.
# 중위값이 1

import numpy as np

from sklearn.datasets import load_boston 

dataset = load_boston()

#1. DATA

x = dataset.data
y = dataset.target # target : x와 y 가 분리한다.


print(x.shape)  # (506, 13) input = 13
print(y.shape)  # (506, )   output = 1
print('==========================================')
print(x[:5])    # 인덱스 0 ~ 4
print(y[:10])   # 인덱스 0 ~ 9

print(np.max(x), np.min(x)) # 최댓값 711.0, 최솟값 0.0   --->  원래는 데이터가 0 ~ 1 사이 값이 되도록 전처리 과정을 거쳐야 함
print(dataset.feature_names) # column 이름

# ********* 데이터 전처리 ( MinMax ) *********
#[1] x를 0 ~ 1로 만들기 위해서 모든 데이터를 최댓값 711 로 나눈다. 
# y는 바꿀 필요 없음
# x = x / 711.     # 소수점으로 만들어 주기 위해서 숫자 뒤에 '.' 을 붙인다.

print(np.max(x[0])) # max = 396.9 ??? 최댓값 711 인데 왜 396 이 나왔을까? ---> 컬럼마다 최솟값과 최댓값이 다르다. 

# [2] 최소가 0인지 몰랐을 때 -----> sklearn에서 수식 제공 중 -----------> MinMaxScaler
# x = (x-최소값) / (최댓값-최소값)
# x = (x - np.min(x)) / (np.max(x)-np.max(x))

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
# scaler = MinMaxScaler()
# scaler = StandardScaler()
scaler = RobustScaler()
scaler.fit(x)
x = scaler.transform(x)

# MinMaxscaler
# print(np.max(x), np.min(x)) # 최댓값 711.0, 최솟값 0.0     ----> 최댓값 1.0 , 최솟값 0.0
# print(np.max(x[0]))         # max = 0.9999999999999999    -----> 컬럼마다 최솟값과 최댓값을 적용해서 구해준다.

# Standardscaler
# print(np.max(x), np.min(x)) # 최댓값 711.0, 최솟값 0.0     ----> 최댓값 9.933930601860268 , -3.9071933049810337
# print(np.max(x[0]))         # max = 0.9999999999999999    -----> 평균 : 0.44105193260704206

# RobustScaler
print(np.max(x), np.min(x)) # 최댓값 711.0, 최솟값 0.0     ----> 최댓값 24.678376790228196 -18.76100251828754
print(np.max(x[0]))         # 1.44

# train_test_split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, shuffle = True, random_state=66)


#2. Modeling
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(128, activation='relu',input_dim=13))
# model.add(Dense(10, activation='relu',input_shape=(13,))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(32))
model.add(Dense(32))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(1))

#3. Compile, Train
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=130, batch_size=8, validation_split=0.1, verbose=1)

#4. Evaluate, Predict
loss, mae = model.evaluate(x_test, y_test, batch_size=8)
print("loss : ", loss)
print("mae : ", mae)

y_predict = model.predict(x_test)
# print("보스턴 집 값 : \n", y_predict)

# RMSE
from sklearn.metrics import mean_squared_error
def RMSE (y_test, y_train) :
    return np.sqrt(mean_squared_error(y_test, y_train))
print("RMSE : ", RMSE(y_test, y_predict))

# R2
from sklearn.metrics import r2_score
R2 = r2_score(y_test, y_predict)
print("R2 : ", R2)

# 전처리 전
# loss :  32.70256042480469
# mae :  4.830300331115723
# RMSE :  5.718615214569833
# R2 :  0.6087411974708226

# 전처리 후 (x = x/711. 일 때) -> 성능 좋아짐
# loss :  12.675890922546387
# mae :  2.6301143169403076
# RMSE :  3.560321795335871
# R2 :  0.8483435532299526

# 전처리 후 (MinMaxScaler 사용했을 때) -> 성능이 더 좋아짐
# loss :  8.257928848266602
# mae :  2.153233766555786
# RMSE :  2.8736613880089967
# R2 :  0.9012007709162857

# StandardScaler 전처리
# loss :  9.856471061706543
# mae :  2.3758113384246826
# RMSE :  3.139501616951642
# R2 :  0.8820755680825718

# RobustScaler 전처리
# loss :  7.069347381591797
# mae :  1.924534559249878
# RMSE :  2.658824540901544
# R2 :  0.915421159330194