# 훈련을 시킬 때 세세한 훈련 과정을 생략해보자 (model.fit)
# verbose = 0 : 완전 생략
# verbose = 1 : [==============================] - 0s 1ms/step loss , mae , val_loss , val_mae 모두 출력, 디폴트값
# verbose = 2 :  0s - loss , mae , val_loss , val_mae: 37.2005
# verbose = 3 이상 : Epoch 2/100 (그 이후 과정 안 보여줌)


import numpy as np

#1 DATA
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

# x_pred2 = np.transpose(x_pred2)   ----> #(5,) 1차원  -----> 행렬 변환
x_pred2 = x_pred2.reshape(1, 5) # [[100, 1, 101, 201, 301]] # inpurt_dim = 5
print("x_pred2.shape_transpose : ", x_pred2.shape)  #(1, 5) 2차원


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66) #3개 행 모두를 행을 기준으로 자른다. #random_state : 랜덤 난수 고정
# print(x_train.shape)      #(80, 5)
# print(y_train.shape)      #(80, 2)
# print(x_test.shape)       #(20, 5)

#2. Modeling
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# from keras.layers import Dense (tesnsorflow 설치가 필요함, 조금 느려짐)

model = Sequential()
model.add(Dense(100, input_dim = 5)) #input 5개
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(2)) # output= 2

#3. Compile, Train
# validation default == None
# (default) verbose = 1 : 훈련되는 과정을 보여준다. > 단점 : 훈련과정을 보여주느라 딜레이가 된다. 
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
print("wait....") #훈련이 돌아가고 있음을 알려주면 마음이 편함
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.2, verbose=0) 
# verbose = 0 : 훈련되는 과정 보여주지 않는다. > 장점 : 시간이 절약된다. 단점 : 과정이 안 보인다.

"""
verbose=0
생략 

verbose=1
Epoch 2/100
64/64 [==============================] - 0s 1ms/step - loss: 11392.2686 - mae: 74.8431 - val_loss: 2371.2507 - val_mae: 40.8041

verbose=2
Epoch 2/100
64/64 - 0s - loss: 2969.9910 - mae: 45.9442 - val_loss: 1758.1270 - val_mae: 37.2005

verbose=3
Epoch 2/100

verbose=4
WARNING:tensorflow:Callbacks method `on_train_batch_end` is slow compared to the batch time (batch time: 0.0000s vs `on_train_batch_end` time: 0.0010s). Check your callbacks.
Epoch 2/100

verbose=5
WARNING:tensorflow:Callbacks method `on_train_batch_end` is slow compared to the batch time (batch time: 0.0000s vs `on_train_batch_end` time: 0.0010s). Check your callbacks.
Epoch 2/100

verbose=6
2020-12-29 15:55:54.633983: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library cublas64_10.dll
Epoch 2/100

verbose=10
WARNING:tensorflow:Callbacks method `on_train_batch_end` is slow compared to the batch time (batch time: 0.0000s vs `on_train_batch_end` time: 0.0010s). Check your callbacks.
"""
#4. Predict, Evaluate
loss, mae = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('mae : : ', mae)

y_predict = model.predict(x_test) 
print(y_predict)
print(y_predict.shape)  #(20,2)

from sklearn.metrics import mean_squared_error #mse
def RMSE (y_test, y_predict) :                 # y_test, y_predict의 shpae 를 맞춰야 함 (20,3)
      return np.sqrt(mean_squared_error(y_test, y_predict)) #RMSE = mse에 루트를 씌운다.
print("RMSE :", RMSE(y_test, y_predict))

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)


y_pred2 = model.predict(x_pred2)
print(y_pred2)  #[[501.50757 572.4987 ]]