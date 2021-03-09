# 입력받는 잇풋 데이터도 다수, 아웃풋도 다수일 때
# 예측하고자 하는 y 값이 여러개일 때
# 주의 : 행렬의 형태(shape)를 통일시켜야 한다. - 특히 '열 ' 중요

# 다 : 다 mlp
# input = 3 , output = 3

import numpy as np

#1 DATA
x = np.array( [range(100), range(301, 401), range(1, 101)] ) # 0 ~ 99 / 301 ~ 400 / 1 ~ 100 --->  (3, 100)
y = np.array( [range(711, 811), range(1, 101), range(201, 301)] )  

print(x.shape)          #(3, 100)
print(y.shape)          #(3, 100)
x = np.transpose(x)     
# print(x)
# print(x.shape)          #(100, 3)
y = np.transpose(y)     
print(y)
print(y.shape)          #(100, 3)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66) #3개 행 모두를 행을 기준으로 자른다. #random_state : 랜덤 난수 고정
print(x_train.shape)      #(80, 3)
print(y_train.shape)      #(80, 3)
print(x_test.shape)       #(20, 3)


# Modeling
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# from keras.layers import Dense (tesnsorflow 설치가 필요함, 조금 느려짐)

model = Sequential()
model.add(Dense(100, input_dim = 3)) #input 3개
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(3)) # output= 3 

# Compile, Train
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.6)  # 첫 번째 컬럼에서 20%, 두 번쩨 컬럼에서 20%, y 컬럼에서 20% # 이때 batch_size=1는 (1,3)을 의미함

# Predict, Evaluate
loss, mae = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('mae : : ', mae)

y_predict = model.predict(x_test)
# print(y_predict)

from sklearn.metrics import mean_squared_error #mse
def RMSE (y_test, y_predict) :                 # y_test, y_predict의 shpae 를 맞춰야 함 (20,3)
      return np.sqrt(mean_squared_error(y_test, y_predict)) #RMSE = mse에 루트를 씌운다.
print("RMSE :", RMSE(y_test, y_predict))

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)
