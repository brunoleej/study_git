# ensemble (1 - 2)
# 모델 병합 : concatenate
# 모델 분기

# 실습
# 1 : 다 앙상블을 구현하시오 

import numpy as np

# Data
x1 = np.array( [range(100), range(301,401), range(1,101)] )         #(3, 100)
y1 = np.array( [range(711, 811), range(1, 101), range(201, 301)] )  

# x2 = np.array([range(101, 201), range(411,511),range(100,200)])
y2 = np.array([range(501, 601), range(711,811), range(100)])

x1 = np.transpose(x1)   #(100, 3)
# x2 = np.transpose(x2)
y1 = np.transpose(y1)
y2 = np.transpose(y2)

from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test = train_test_split (x1, y1, shuffle=False, train_size=0.8)
y2_train, y2_test = train_test_split (y2, shuffle=False, train_size=0.8)

#Modeling
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

# 두 모델을 합쳤다가 다시 분리하는 과정

# Model 1
input1 = Input(shape=(3,)) #input = 3
dense1 = Dense(100, activation = 'relu')(input1)
dense1 = Dense(40, activation = 'relu')(dense1)
dense1 = Dense(10, activation = 'relu')(dense1)
dense1 = Dense(10, activation = 'relu')(dense1)
dense1 = Dense(10, activation = 'relu')(dense1)
dense1 = Dense(5, activation = 'relu')(dense1)
# output1 = Dense(3)(dense1)

# Model 2
# input2 = Input(shape=(3,))  #input = 3
# dense2 = Dense(10, activation = 'relu')(input2)
# dense2 = Dense(5, activation = 'relu')(dense2)
# dense2 = Dense(5, activation = 'relu')(dense2)
# dense2 = Dense(5, activation = 'relu')(dense2)
# output2 = Dense(3)(dense2)

# 모델 병합 : concatenate
# model1과 model2가 merge하면서 서로의 가중치를 공유한다. (각 모델이 서로에게 영향을 미친다.)

# from tensorflow.keras.layers import concatenate, Concatenate
# from keras.layers.merge import concatenate, Concatenate
# from keras.layers import concatenate, Concatenate

# merge도 layers에 속해있으므로 layer를 구성한다.
# merge1 = concatenate([dense1, dense2]) # 두 모델의 마지막 층에 있는 레이어를 합친다.
# middle1 = Dense(30)(merge1)
# middle1 = Dense(10)(middle1)
# middle1 = Dense(10)(middle1)
# middle1 = Dense(10)(middle1)

# 둘로 합쳤던 것을 다시 나눈다. merge의 마지막 층을 가져온다.
# 모델 분기 1 
output1 = Dense(100)(dense1)
output1 = Dense(100)(output1)
output1 = Dense(10)(output1)
output1 = Dense(10)(output1)
output1 = Dense(10)(output1)
output1 = Dense(3)(output1) # y1 :output = 3

# 모델 분기 2
output2 = Dense(100)(dense1)
output2 = Dense(100)(output2)
output2 = Dense(50)(output2)
output2 = Dense(50)(output2)
output2 = Dense(10)(output2)
output2 = Dense(10)(output2)
output2 = Dense(3)(output2) # y2 :output = 3

# 모델 선언
# 최종적인 input, output을 넣어서 모델 구성
# 두 개 이상은 리스트로 묶는다.
model = Model(inputs = input1, outputs = [output1, output2])
model.summary()


# Compile, fit
# 두 개 이상은 리스트로 묶는다.

model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit (x1_train, [y1_train, y2_train], \
            epochs=100, batch_size=1, validation_split=0.2, verbose=1)

# Evaluate, Perdict
loss = model.evaluate(x1_test, [y1_test, y2_test], batch_size=1)
print("loss : ", loss) 

# loss, mae :  [3749.34814453125, 1920.6214599609375, 1828.7269287109375, 40.63865280151367, 40.203800201416016]
#              [대표 loss(첫 번째loss + 두 번째loss) , 첫 번째 모델의 loss, 두 번째 모델의 loss, 첫 번째 모델의 metrics, 두 번째 모델의 metrics ]


# 위의 출력값과 연결
print("model.metrics_names : ", model.metrics_names)
# model.metrics_names :  ['loss', 'dense_7_loss', 'dense_12_loss', 'dense_7_mae', 'dense_12_mae']

y1_predict, y2_predict = model.predict(x1_test)
print("================================")
print("y1_predict : \n", y1_predict)        #(20,3)
print("================================")
print("y2_predict : \n", y2_predict)        #(20,3)
print("================================")

# RMSE 
from sklearn.metrics import mean_squared_error #mse
def RMSE (y_test, y_predict) :                 
      return np.sqrt(mean_squared_error(y_test, y_predict)) #RMSE = mse에 루트를 씌운다.

RMSE1 = RMSE(y1_test, y1_predict)
RMSE2 = RMSE(y2_test, y2_predict)
RMSE = (RMSE1 + RMSE2)/2 #전체 RMSE
print("RMSE1 : ", RMSE1)
print("RMSE2 : ", RMSE2)
print("RMSE : ", RMSE)

# R2
from sklearn.metrics import r2_score
r2_1 = r2_score(y1_test, y1_predict)
r2_2 = r2_score(y2_test, y2_predict)
r2 = (r2_1 + r2_2) / 2  #전체 r2
print("R2_1 : ", r2_1)
print("R2_2 : ", r2_2)
print("R2 : ", r2)

# RMSE1 :  32.88857473121615
# RMSE2 :  36.02375015199405
# RMSE :  34.4561624416051

# R2_1 :  -31.531078130850826
# R2_2 :  -38.02888947408395
# R2 :  -34.779983802467385