# 다 : 1 mlp 모델을 구성하시오
# MinMaxScalar
# [1] 데이터 전처리 MinMaxScaler : 데이터를 모두 0과 1사이로 바꾼다.
# 최솟값이 0일 때, 모든 x의 값을 최댓값으로 바꿔준다.

import numpy as np
from sklearn.datasets import load_boston 

# Data
boston = load_boston()
x = boston.data
y = boston.target # target : x와 y 가 분리한다.

print(x.shape)  # (506, 13) input = 13
print(y.shape)  # (506, )   output = 1
print('==========================================')
print(x[:5])    # 인덱스 0 ~ 4
print(y[:10])   # 인덱스 0 ~ 9
print(np.max(x), np.min(x)) # 최댓값 711.0, 최솟값 0.0   --->  원래는 데이터가 0 ~ 1 사이 값이 되도록 전처리 과정을 거쳐야 함
print(boston.feature_names) # column 이름

# ********* 데이터 전처리 ( MinMax ) *********
# x를 0 ~ 1로 만들기 위해서 모든 데이터를 최댓값 711 로 나눈다. 
# y는 바꿀 필요 없음
x = x / 711.     # 소수점으로 만들어 주기 위해서 숫자 뒤에 '.' 을 붙인다.

# print(boston.DESCR)         # column 이름의 의미
'''
    :Attribute Information (in order):
        - CRIM     per capita crime rate by town
        - ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
        - INDUS    proportion of non-retail business acres per town
        - CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
        - NOX      nitric oxides concentration (parts per 10 million)
        - RM       average number of rooms per dwelling
        - AGE      proportion of owner-occupied units built prior to 1940
        - DIS      weighted distances to five Boston employment centres
        - RAD      index of accessibility to radial highways
        - TAX      full-value property-tax rate per $10,000
        - PTRATIO  pupil-teacher ratio by town
        - B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
        - LSTAT    % lower status of the population
        - MEDV     Median value of owner-occupied homes in $1000's
'''

# train_test_split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, shuffle = True, random_state=66)

# Modeling
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(128,activation = 'relu',input_dim=13),
    # model.add(Dense(10, activation='relu',input_shape=(13,))
    Dense(128),
    Dense(64),
    Dense(64),
    Dense(32),
    Dense(32),
    Dense(16),
    Dense(16),
    Dense(8),
    Dense(1)
])

# Compile, fit
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=150, batch_size=8, validation_split=0.1, verbose=1)

# Evaluate 
loss, mae = model.evaluate(x_test, y_test, batch_size=8)
print("loss : ", loss)
print("mae : ", mae)

# Prediction
y_predict = model.predict(x_test)
# print("y_pred : \n", y_predict)

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

# 전처리 후
# loss :  12.675890922546387
# mae :  2.6301143169403076
# RMSE :  3.560321795335871
# R2 :  0.8483435532299526