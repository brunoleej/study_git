# ensemble ( 2 - 1 - 3 )
# 모델 병합 : concatenate
# 모델 분기

# 실습 
# 다 : 다 앙상블을 구현하시오 

import numpy as np

# Data
x1 = np.array( [range(100), range(301,401), range(1,101)] )         #(3, 100)
y1 = np.array( [range(711, 811), range(1, 101), range(201, 301)] )  

x2 = np.array([range(101, 201), range(411,511),range(100,200)])
y2 = np.array([range(501, 601), range(711,811), range(100)])

y3 = np.array([range(601, 701), range(811,911), range(1100,1200)])

x1 = np.transpose(x1)   #(100, 3)
x2 = np.transpose(x2)
y1 = np.transpose(y1)
y2 = np.transpose(y2)
y3 = np.transpose(y3)

from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test = train_test_split (x1, y1, shuffle=False, train_size=0.8)
x2_train, x2_test, y2_train, y2_test = train_test_split (x2, y2, shuffle=False, train_size=0.8)
y3_train, y3_test = train_test_split (y3, shuffle=False, train_size=0.8)

# Modeling
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

# 두 모델을 합쳤다가 다시 분리하는 과정
# Model 1
input1 = Input(shape=(3,)) #input = 3
dense1 = Dense(100, activation = 'relu')(input1)
dense1 = Dense(100, activation = 'relu')(dense1)
dense1 = Dense(10, activation = 'relu')(dense1)
dense1 = Dense(10, activation = 'relu')(dense1)
dense1 = Dense(10, activation = 'relu')(dense1)
# output1 = Dense(3)(dense1)

# Model 2
input2 = Input(shape=(3,))  #input = 3
dense2 = Dense(100, activation = 'relu')(input2)
dense2 = Dense(100, activation = 'relu')(dense2)
dense2 = Dense(10, activation = 'relu')(dense2)
dense2 = Dense(10, activation = 'relu')(dense2)
dense2 = Dense(10, activation = 'relu')(dense2)
# output2 = Dense(3)(dense2)

# 모델 병합 : concatenate
from tensorflow.keras.layers import concatenate, Concatenate

merge1 = concatenate([dense1, dense2]) # 두 모델의 마지막 층에 있는 레이어를 합친다.
middle1 = Dense(100)(merge1)
middle1 = Dense(100)(middle1)
middle1 = Dense(10)(middle1)
middle1 = Dense(10)(middle1)

# 둘로 합쳤던 것을 다시 나눈다. merge의 마지막 층을 가져온다.
# 모델 분기 1 
output1 = Dense(50)(middle1)
output1 = Dense(40)(output1)
output1 = Dense(20)(output1)
output1 = Dense(20)(output1)
output1 = Dense(10)(output1)
output1 = Dense(3)(output1) # y1 :output = 3

# 모델 분기 2
output2 = Dense(40)(middle1)
output2 = Dense(40)(middle1)
output2 = Dense(10)(output2)
output2 = Dense(10)(output2)
output2 = Dense(10)(output2)
output2 = Dense(3)(output2) # y2 :output = 3

# 모델 분기 3
output3 = Dense(40)(middle1)
output3 = Dense(20)(middle1)
output3 = Dense(20)(output3)
output3 = Dense(10)(output3)
output3 = Dense(10)(output3)
output3 = Dense(3)(output3) # y3 :output = 3

# 모델 선언 (뒤에서 한다.)
# 최종적인 input, output을 넣어서 모델 구성
model = Model(inputs = [input1, input2], outputs = [output1, output2, output3])
model.summary()

# Compile, Train

model.compile(loss='mse', optimizer='adam', metrics=['mae'])
print("please wait....")
model.fit ([x1_train, x2_train], [y1_train, y2_train, y3_train], \
            epochs=110, batch_size=1, validation_split=0.2, verbose=0)

# Evaluate, Perdict
loss = model.evaluate([x1_test,x2_test], [y1_test, y2_test, y3_test], batch_size=1)
print("loss : ", loss) 

# loss, mae :  [4118.6650390625, 1789.0152587890625, 1717.032470703125, 612.617431640625, 36.35871887207031, 30.847341537475586, 22.343196868896484]
#              [대표 loss(첫 번째loss + 두 번째loss) , 첫 번째 모델의 loss, 두 번째 모델의 loss, 세 번째 모델의 loss, 첫 번째 모델의 metrics, 두 번째 모델의 metrics, 세 번째 모델의 metrics]


# 위의 출력값과 연결
print("model.metrics_names : ", model.metrics_names)
# model.metrics_names :   ['loss', 'dense_12_loss', 'dense_17_loss', 'dense_22_loss', 'dense_12_mae', 'dense_17_mae', 'dense_22_mae']

y1_predict, y2_predict, y3_predict = model.predict([x1_test, x2_test])
print("================================")
print("y1_predict : \n", y1_predict)        #(20,3)
print("================================")
print("y2_predict : \n", y2_predict)        #(20,3)
print("================================")
print("y3_predict : \n", y3_predict)        #(20,3)
print("================================")

# RMSE 
from sklearn.metrics import mean_squared_error #mse
def RMSE (y_test, y_predict) :                 
      return np.sqrt(mean_squared_error(y_test, y_predict)) #RMSE = mse에 루트를 씌운다.
# 
RMSE1 = RMSE(y1_test, y1_predict)
RMSE2 = RMSE(y2_test, y2_predict)
RMSE3 = RMSE(y3_test, y3_predict)
RMSE = (RMSE1 + RMSE2 + RMSE3)/3 #전체 RMSE
print("RMSE1 : ", RMSE1)
print("RMSE2 : ", RMSE2)
print("RMSE3 : ", RMSE3)
print("RMSE : ", RMSE)

# R2
from sklearn.metrics import r2_score
r2_1 = r2_score(y1_test, y1_predict)
r2_2 = r2_score(y2_test, y2_predict)
r2_3 = r2_score(y3_test, y3_predict)
r2 = (r2_1 + r2_2 + r2_3) / 3  #전체 r2
print("R2_1 : ", r2_1)
print("R2_2 : ", r2_2)
print("R2_3 : ", r2_3)
print("R2 : ", r2)

# RMSE1 :  2.825380974771093
# RMSE2 :  2.916157260204751
# RMSE3 :  3.5001077045035927
# RMSE :  3.0805486464931455

# R2_1 :  0.7599164615759865
# R2_2 :  0.7442414085339886
# R2_3 :  0.6315562723872058
# R2 :  0.711904714165727