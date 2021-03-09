# Multi Layer Percemtron (MLP)
# [1] 인풋을 1개가 아닌 2개 이상을 받는 경우
# input_dim = 2
# [2] 행렬의 행과 렬을 교환한다.
# x = np.transpose(x)


# 다 : 1 mlp
# input = 2 , output = 1

import numpy as np

#1 DATA
# x = np.array([1,2,3,4,5,6,7,8,9,10]) # 스칼라가 10개인 벡터 x ---> (10,)
x = np.array([[1,2,3,4,5,6,7,8,9,10],
            [11,12,13,14,15,16,17,18,19,20]]) # 스칼라가 10개인 벡터 x ---> (2, 10)  ---> (10,2) 로 바꾸고 싶다!!!
y = np.array([1,2,3,4,5,6,7,8,9,10]) 

print(x.shape)          #(10,) - 스칼라가 10개라는 의미 --> (2, 10)

# 행렬 (2,10)을 (10,2)로 바꾸어라

#[1] 
x = np.transpose(x)     
print(x)
print(x.shape)          #(10, 2)
"""
  x1 x2    # 10행 2열
[[ 1 11]
 [ 2 12]
 [ 3 13]
 [ 4 14]
 [ 5 15]
 [ 6 16]
 [ 7 17]
 [ 8 18]
 [ 9 19]
 [10 20]]
"""
#[2]
# xx = x.T
# print(xx)

#[3]
# xx = np.swapaxes(x, 0 ,1)
# print(xx.shape)

#2. Modeling
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# from keras.layers import Dense (tesnsorflow 설치가 필요함, 조금 느려짐)

model = Sequential()
model.add(Dense(10, input_dim = 2)) #input 2개
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(1))

#3. Compile, Train
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x, y, epochs=100, batch_size=1, validation_split=0.2)  # 첫 번째 컬럼에서 20%, 두 번쩨 컬럼에서 20%, y 컬럼에서 20%

#4. Predict, Evaluate
loss, mae = model.evaluate(x, y)
print('loss : ', loss)
print('mae : : ', mae)

y_predict = model.predict(x)
# print(y_predict)


# from sklearn.metrics import mean_squared_error #mse
# def RMSE (y_test, y_predict) :
#       return np.sqrt(mean_squared_error(y_test, y_predict)) #RMSE = mse에 루트를 씌운다.
# print("RMSE :", RMSE(y_test, y_predict))

# from sklearn.metrics import r2_score
# r2 = r2_score(y_test, y_predict)
# print("R2 : ", r2)