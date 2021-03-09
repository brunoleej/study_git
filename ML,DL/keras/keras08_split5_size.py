# 실습 : train_size, test_size를 구분 : 1을 넘어간 경우, 1이 안되는 경우
# 각 파라미터 별로 어떤 결과가 나오는지 확인한다.

#1이 넘어가는 경우 에러 발생

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import numpy as np

#1. DATA

x = np.array(range(1,11)) #1부터 10까지
y = np.array(range(1,11)) #1부터 10까지


# sklearn을 활용한다.
from sklearn.model_selection import train_test_split

# [1] x_train , x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=False) #x, y 데이터 중 80%를 train으로 준다. / 20% 는 테스트로 준다.
# [2] 
x_train , x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, test_size=0.2, shuffle=False) 
# [3] x_train , x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, test_size=0.2, shuffle=False) 
# [4] x_train , x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, test_size=0.2, shuffle=False) 

print("x_train : ", x_train) 
print("x_test : ", x_test)
print("x_train 크기 : ", x_train.shape) #(80,) >> 스칼라 60개 , 1차원
print("x_test 크기 : ", x_test.shape)  #(20,) 
print("y_train 크기 : ", y_train.shape) #(80,)
print("y_test 크기 : ", y_test.shape)  #(20,)


"""
[1] x_train , x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=False)
x_train :  [1 2 3 4 5 6 7 8]
x_test :  [ 9 10]
x_train 크기 :  (8,)
x_test 크기 :  (2,)
y_train 크기 :  (8,)
y_test 크기 :  (2,)

[2]
x_train :  [1 2 3 4 5 6 7 8]
x_test :  [ 9 10]
x_train 크기 :  (8,)
x_test 크기 :  (2,)
y_train 크기 :  (8,)
y_test 크기 :  (2,)

[3]
ValueError: The sum of test_size and train_size = 1.1, should be in the (0, 1) range. Reduce test_size and/or train_size.

[4]
x_train :  [1 2 3 4 5 6 7]
x_test :  [8 9]
x_train 크기 :  (7,)
x_test 크기 :  (2,)
y_train 크기 :  (7,)
y_test 크기 :  (2,)
"""

"""
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, \
                                                   test_size=0.2, shuffle=False) #train 데이터를 train, validation 각각 80%, 20%로 나눈다.
# x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, \
#                                                    train_size=0.8, shuffle=False) #train 데이터를 train, validation 각각 80%, 20%로 나눈다.                                                   
print(x_train)
print(x_val)

print(x_train.shape) #(64,)
print(x_val.shape)   #(16,) 
print(y_train.shape) #(64,)
print(y_val.shape)   #(16,)

# train 64개 / val 16개 / test 20개

#2. Modeling
model = Sequential()
model.add(Dense(100, input_dim=1)) # 기본값 : activation='linear' 
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

#3. Compile, Train
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=100, validation_data=(x_val, y_val)) # validation 16개

#4. Evaluate, Predict
loss, mae = model.evaluate(x_test,y_test)
print("loss : ", loss)
print("mae : ", mae)

y_predict = model.predict(x_test)
print(y_predict)

# 사이킷런(sklearn) 설치
from sklearn.metrics import mean_squared_error #mse
def RMSE (y_test, y_predict) :
      return np.sqrt(mean_squared_error(y_test, y_predict)) #RMSE = mse에 루트를 씌운다.
print("RMSE :", RMSE(y_test, y_predict))

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)

# shuffle = false
# loss :  0.011699133552610874
# mae :  0.10681991279125214

# shuffle = True >> True일 때 결과가 더 좋다.
# loss :  0.0076343403197824955
# mae :  0.06808438152074814

# validation_split = 0.2 >> validation을 넣었더니 성능이 더 떨어졌다. >> 왜? 훈련량 자체가 적어졌기 때문에
# loss :  0.0492154136300087
# mae :  0.174998477101326

# train, val, test 구분
# loss :  0.002284829970449209
# mae :  0.04123806953430176
# RMSE : 0.04779989683543168
# R2 :  0.9999969615004289
"""