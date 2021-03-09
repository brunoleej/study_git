# 실습 train과 test data를 분리해서 소스를 완성하시오
# train_test_split 사용
# random_state

# 다 : 1 mlp
# input = 3 , output = 1
 

import numpy as np

# DATA
x = np.array( [range(100), range(301, 401), range(1, 101)] ) # 0 ~ 99 / 301 ~ 400 / 1 ~ 100 --->  (3, 100)
y = np.array(range(711, 811))                                #710 ~ 810 ----> (100, )
# print(x.shape)          #(3, 100)
# print(y.shape)          #(100, )

# 행렬(3, 100)을 (100,3)로 바꾸어라
x = np.transpose(x)     
# print(x)
# print(x.shape)          #(100, 3)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66) #3개 행 모두를 행을 기준으로 자른다.
#random_state : 랜덤 난수 고정 (난수표의 위치 : 66) ---> 66번 위치에 있는 난수로만 계속 난수를 생성한다. ---> 일정한 난수를 생성할 수 있음

print(x_train.shape)      #(80, 3)
print(y_train.shape)      #(80,)
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
model.add(Dense(1))

# Compile, Train
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.6)  

#4. Predict, Evaluate
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


# loss :  1.6763805898989403e-09
# mae : :  2.746581958490424e-05
# RMSE : 4.0943627517696345e-05
# R2 :  0.9999999999978789

# 데이터 x1, x2, x3 모두 w = 1 >>>> 결과가 좋게 나올 수 밖에 없다.