# 모델
#[**1] Model()
#[2] Sequential()
import numpy as np

# DATA
x = np.array( [range(100), range(1, 101), range(101,201), range(201, 301), range(301, 401)] ) 
y = np.array([range(511,611), range(611,711)])  
# print(x.shape)          #(5, 100)
# print(y.shape)          #(2, 100)

x_pred2 = np.array([100, 1, 101, 201, 301])
print("x_pred2.shape : ", x_pred2.shape) # ----> (5,) 스칼라, 1차원, inpurt_dim = 1

x = np.transpose(x)     
# print(x)
# print(x.shape)          #(100, 5)
y = np.transpose(y)     
# print(y)
# print(y.shape)          #(100, 2)

x_pred2 = x_pred2.reshape(1, 5) # [[100, 1, 101, 201, 301]] # inpurt_dim = 5
print("x_pred2.shape_transpose : ", x_pred2.shape)  #(1, 5) 2차원

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66) #3개 행 모두를 행을 기준으로 자른다. #random_state : 랜덤 난수 고정
# print(x_train.shape)      #(80, 5)
# print(y_train.shape)      #(80, 2)
# print(x_test.shape)       #(20, 5)

#2. Modeling
from tensorflow.keras.models import Sequential, Model #함수형 모델, 모델을 재사용할 때 사용
from tensorflow.keras.layers import Dense, Input

#[1] 과 [2] 동일함

#[1] Model() 함수형 구성 : 재사용하는데 용이하다.
input1 = Input(shape = (5,)) #input layer 
dense1 = Dense(5, activation='relu')(input1)
dense2 = Dense(3)(dense1)
dense3 = Dense(4)(dense2)
outputs = Dense(2)(dense3)
model = Model(inputs = input1, outputs = outputs) #모델 선언을 마지막에 한다. :  input layer ~ output layer
model.summary()
'''
Model: "functional_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         [(None, 1)]               0
_________________________________________________________________
dense (Dense)                (None, 5)                 10
_________________________________________________________________
dense_1 (Dense)              (None, 3)                 18
_________________________________________________________________
dense_2 (Dense)              (None, 4)                 16
_________________________________________________________________
dense_3 (Dense)              (None, 1)                 5
=================================================================
Total params: 49
'''

#[2]  Sequential()
# model = Sequential()
# # model.add(Dense(5, input_dim = 1)) #input 5개
# model.add(Dense(5, activation='relu',input_shape = (5,)))  #input layer 
# model.add(Dense(3))
# model.add(Dense(4))
# model.add(Dense(2)) 
# model.summary() 
'''
Model: "sequential"
=================================================================
dense (Dense)                (None, 5)                 10
_________________________________________________________________
dense_1 (Dense)              (None, 3)                 18
_________________________________________________________________
dense_2 (Dense)              (None, 4)                 16
_________________________________________________________________
dense_3 (Dense)              (None, 1)                 5
=================================================================
Total params: 49
'''



#3. Compile, Train
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.2, verbose=0)  # 첫 번째 컬럼에서 20%, 두 번쩨 컬럼에서 20%, y 컬럼에서 20% # 이때 batch_size=1는 (1,3)을 의미함

#4. Predict, Evaluate
loss, mae = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('mae : : ', mae)

y_predict = model.predict(x_test) 
print(y_predict)
print(y_predict.shape)  #(20,2)

from sklearn.metrics import mean_squared_error #mse
def RMSE (y_test, y_predict) :                 # y_test, y_predict의 shpae 를 맞춰야 함 (20,2)
      return np.sqrt(mean_squared_error(y_test, y_predict)) 
print("RMSE :", RMSE(y_test, y_predict))

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)


y_pred2 = model.predict(x_pred2)
print(y_pred2)  # [[561.34485 661.3729 ]]
