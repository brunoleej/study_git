# 실습 : validation_data 를 만들 것
# train_test_split을 사용할 것

# train_test_split(x, y, train_size=0.8, shuffle=True)한 후, 
# x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, shuffle=False) 한 번 더 쪼갠다.
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import numpy as np

#1. DATA
x = np.array(range(1,101)) #1부터 100까지
y = np.array(range(1,101)) #101부터 200까지

# sklearn을 활용한다.
from sklearn.model_selection import train_test_split
# x_train , x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True) #x, y 데이터 중 80%를 train으로 준다. / 20% 는 테스트로 준다.
x_train , x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True) #위와 동일
print(x_train) 
""" x_train 출력결과 (무작위 출력)
[ 13   7  83  89  38  61   5  55  31  46  97  50  29  17  85  67  37 100
  62  79  25  21  16  40  90  58  70  94  91  27  69  66  99  36  20  51
  59  11  28  57  53  80  81  64   4  18  73  43  56  22  52  26  33  39
  42  41  12  60  98  19]
"""
print(x_train.shape) #(80,) >> 스칼라 60개 , 1차원
print(x_test.shape)  #(20,) 
print(y_train.shape) #(80,)
print(y_test.shape)  #(20,)

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, \
                                                   test_size=0.2, shuffle=False) #train 데이터를 train, validation 각각 80%, 20%로 나눈다.
# x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, \
#                                                    train_size=0.8, shuffle=False) #위와 동일                                                   
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

#=======train data, validation 구분==================
# shuffle = True, True
# loss :  0.0032686027698218822
# mae :  0.0511455163359642

# shuffle = True, False
# loss :  0.004108855966478586
# mae :  0.05241196230053902

# shuffle = False, True
# loss :  0.01101187989115715
# mae :  0.10380172729492188

# shuffle = False, False
# loss :  0.012333774007856846
# mae :  0.1102726012468338
