# ensemble ( 2 - 1 - 1)
# 실습 
# 다 : 1 앙상블을 구현하시오 (y2 제거, 분기하는 부분을 뺀다.)
import numpy as np

# Data
x1 = np.array([range(100), range(301,401), range(1,101)])         #(3, 100)
x2 = np.array([range(101, 201), range(411,511),range(100,200)])

y1 = np.array( [range(711, 811), range(1, 101), range(201, 301)] )  
# y2 = np.array([range(501, 601), range(711,811), range(100)])

x1 = np.transpose(x1)   #(100, 3)
x2 = np.transpose(x2)
y1 = np.transpose(y1)
# y2 = np.transpose(y2)

from sklearn.model_selection import train_test_split
# # [1]
# x1_train, x1_test, y1_train, y1_test = train_test_split (x1, y1, shuffle=False, train_size=0.8)
# x2_train, x2_test = train_test_split (x2, shuffle=False, train_size=0.8)
# # [2]
# x1_train, x1_test, y1_train, y1_test = train_test_split (x1, y1, shuffle=False, train_size=0.8)
# x2_train, x2_test, y1_train, y1_test = train_test_split (x2, y1, shuffle=False, train_size=0.8)
# [3]
x1_train, x1_test, x2_train, x2_test, y1_train, y1_test = train_test_split (x1, x2, y1, shuffle=False, train_size=0.8)

print(x1_train.shape)   #(80, 3)
print(x2_test.shape)    #(20, 3)
print(y1_train.shape)   #(80, 3)

# Modeling
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

# Model 1
input1 = Input(shape=(3,)) #input = 3
dense1 = Dense(10, activation = 'relu')(input1)
dense1 = Dense(5, activation = 'relu')(dense1)
# output1 = Dense(3)(dense1)

# Model 2
input2 = Input(shape=(3,))  #input = 3
dense2 = Dense(10, activation = 'relu')(input2)
dense2 = Dense(5, activation = 'relu')(dense2)
dense2 = Dense(5, activation = 'relu')(dense2)
dense2 = Dense(5, activation = 'relu')(dense2)
# output2 = Dense(3)(dense2)

# 모델 병합 : concatenate
from tensorflow.keras.layers import concatenate, Concatenate

# merge도 layers에 속해있으므로 layer를 구성한다.
merge1 = concatenate([dense1, dense2]) # 두 모델의 마지막 층에 있는 레이어를 합친다.
middle1 = Dense(30)(merge1)
middle1 = Dense(10)(middle1)
middle1 = Dense(10)(middle1)
middle1 = Dense(10)(middle1)

# merge의 마지막 층을 가져온다. (둘로 나누지 않는다.)
output1 = Dense(30)(middle1)
output1 = Dense(7)(output1)
output1 = Dense(3)(output1) # y1 :output = 3 (마지막 아웃풋)

# 모델 분기 2
# output2 = Dense(15)(middle1)
# output2 = Dense(7)(output2)
# output2 = Dense(7)(output2)
# output2 = Dense(7)(output2)
# output2 = Dense(3)(output2) # y2 :output = 3

# 모델 선언 (뒤에서 한다.)
# 최종적인 input, output을 넣어서 모델 구성
model = Model(inputs = [input1, input2], outputs = output1)
model.summary()


# Compile, fit
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit ([x1_train, x2_train], y1_train, \
            epochs=10, batch_size=1, validation_split=0.2, verbose=1)

# Evaluate, Perdict
loss = model.evaluate([x1_test,x2_test], y1_test, batch_size=1)
print("loss : ", loss) 

# loss, mae :  [1050.37060546875, 27.982452392578125]
#              ['loss', 'mae']

# 위의 출력값과 연결
print("model.metrics_names : ", model.metrics_names)
# model.metrics_names :  ['loss', 'mae']

y1_predict  = model.predict([x1_test, x2_test])
print("================================")
print("y1_predict : \n", y1_predict)        #(20,3)
print("================================")
# print("y2_predict : \n", y2_predict)        #(20,3)
# print("================================")

# RMSE 
from sklearn.metrics import mean_squared_error #mse
def RMSE (y_test, y_predict) :                 
      return np.sqrt(mean_squared_error(y_test, y_predict)) #RMSE = mse에 루트를 씌운다.

RMSE = RMSE(y1_test, y1_predict)
# RMSE2 = RMSE(y2_test, y2_predict)
# RMSE = (RMSE1 + RMSE2)/2 #전체 RMSE
print("RMSE : ", RMSE)
# print("RMSE2 : ", RMSE2)
# print("RMSE : ", RMSE)

# R2
from sklearn.metrics import r2_score
r2 = r2_score(y1_test, y1_predict)
# r2_2 = r2_score(y2_test, y2_predict)
# r2 = (r2_1 + r2_2) / 2  #전체 r2
print("R2 : ", r2)
# print("R2_2 : ", r2_2)
# print("R2 : ", r2)

# RMSE :  42.24065529123472
# R2 :  -52.66234464459896
