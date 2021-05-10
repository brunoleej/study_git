# 다차원의 데이터를 입력시키기 위해서 input_dim=5 대신 input_shape=(5,) 사용한다.
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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
# model.add(Dense(100, input_dim = 5)) #input 5개
model.add(Dense(100, input_shape = (5,))) 
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(2)) # output= 2

model.summary()
#3. Compile, Train
# validation default == None
# (default) verbose = 1 : 훈련되는 과정을 보여준다. > 단점 : 훈련과정을 보여주느라 딜레이가 된다. 
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
      return np.sqrt(mean_squared_error(y_test, y_predict)) #RMSE = mse에 루트를 씌운다.
print("RMSE :", RMSE(y_test, y_predict))

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)

y_pred2 = model.predict(x_pred2)
print(y_pred2)  #[[485.261  542.4673]]