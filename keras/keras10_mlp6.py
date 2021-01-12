# input = 1 , output = 3 일 때, 모델 형성 가능하다.
# mlp4 처럼 x_predict만들어서 예측

# 1 : 다 mlp
# input = 1 , output = 3

import numpy as np

#1 DATA
x = np.array( range(100) )
y = np.array( [range(711, 811), range(1, 101), range(201, 301)] )  

print(x.shape)          #(100, )
print(y.shape)          #(3, 100)
x = np.transpose(x)     
# print(x.shape)          #(100, )
y = np.transpose(y)     
# print(y.shape)            #(100, 3)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66) #3개 행 모두를 행을 기준으로 자른다. #random_state : 랜덤 난수 고정
# print(x_train.shape)     #(80,) 
# print(y_train.shape)     #(80,3)
# print(x_test.shape)      #(20,)

x_pred2 = np.array([100])
x_pred2 = x_pred2.reshape(1,1) 
# print("x_pred2.shape_transpose : ", x_pred2.shape)  #(1, 1) 2차원


#2. Modeling
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(100, input_dim = 1)) #input 1개
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(3)) # output= 3 

#3. Compile, Train
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.6)  # 첫 번째 컬럼에서 20%, 두 번쩨 컬럼에서 20%, y 컬럼에서 20% # 이때 batch_size=1는 (1,3)을 의미함

#4. Predict, Evaluate
loss, mae = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('mae : : ', mae)

y_predict = model.predict(x_test)
print(y_predict) 

from sklearn.metrics import mean_squared_error #mse
def RMSE (y_test, y_predict) :                 # y_test, y_predict의 shpae 를 맞춰야 함 (20,3)
      return np.sqrt(mean_squared_error(y_test, y_predict)) #RMSE = mse에 루트를 씌운다.
print("RMSE :", RMSE(y_test, y_predict))

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)

y_predict2 = model.predict(x_pred2)
print(y_predict2)       #[[810.65704 100.70104 301.36755]]
print(y_predict2.shape) #(1,3)