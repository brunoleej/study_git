# metrics가 다를 경우, 
# metrics가 여러 개인 경우 metrics = ['mse','mae'] 처럼 리스트로 구현한다.

import numpy as np

# Data
x1 = np.array( [range(100), range(301,401), range(1,101)] )         #(3, 100)
y1 = np.array( [range(711, 811), range(1, 101), range(201, 301)] )  

x2 = np.array([range(101, 201), range(411,511),range(100,200)])
y2 = np.array([range(501, 601), range(711,811), range(100)])

x1 = np.transpose(x1)   #(100, 3)
x2 = np.transpose(x2)
y1 = np.transpose(y1)   #(100, 3)
y2 = np.transpose(y2)

from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test = train_test_split (x1, y1, shuffle=False, train_size=0.8)
x2_train, x2_test, y2_train, y2_test = train_test_split (x2, y2, shuffle=False, train_size=0.8)

# train (80, 3)
# test (20, 3)

#2. Modeling

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

# 두 모델을 합쳤다가 다시 분리하는 과정

# Model 1
input1 = Input(shape=(3,))    # input : (3,)
dense1 = Dense(10, activation = 'relu')(input1)
dense1 = Dense(5, activation = 'relu')(dense1)

# Model 2
input2 = Input(shape=(3,))    # input : (3,)
dense2 = Dense(10, activation = 'relu')(input2)
dense2 = Dense(5, activation = 'relu')(dense2)
dense2 = Dense(5, activation = 'relu')(dense2)
dense2 = Dense(5, activation = 'relu')(dense2)

# Model concatenate
from tensorflow.keras.layers import concatenate, Concatenate

# merge도 layers에 속해있으므로 layer를 구성한다.
merge1 = concatenate([dense1, dense2]) # 두 모델의 마지막 층에 있는 레이어를 합친다.
middle1 = Dense(30)(merge1)
middle1 = Dense(10)(middle1)
middle1 = Dense(10)(middle1)
middle1 = Dense(10)(middle1)

# 모델 분기 1 
output1 = Dense(30)(middle1)
output1 = Dense(7)(output1)
output1 = Dense(3)(output1) # 최종 output = 3

# 모델 분기 2
output2 = Dense(15)(middle1)
output2 = Dense(7)(output2)
output2 = Dense(7)(output2)
output2 = Dense(7)(output2)
output2 = Dense(3)(output2) # 최종 output = 3

# 모델 선언
model = Model(inputs = [input1, input2], outputs = [output1, output2])
model.summary()

# Compile, fit

model.compile(loss='mse', optimizer='adam', metrics=['mae', 'mse'])
model.fit ([x1_train, x2_train], [y1_train, y2_train], \
            epochs=10, batch_size=1, validation_split=0.2, verbose=1)

#4. Evaluate, Perdict
loss = model.evaluate([x1_test,x2_test], [y1_test, y2_test], batch_size=1)
print("loss : ", loss) 

# loss, mae, mse :  [1196.434814453125, 556.9600830078125, 639.4747314453125, 21.187952041625977, 556.9600830078125, 17.66719627380371, 639.4747314453125]
#                   [대표 loss(첫 번째loss + 두 번째loss) , 첫 번째 모델의 loss, 두 번째 모델의 loss, 첫 번째 모델의 mae, 첫 번째 모델의 mse, 두 번째 모델의 mae, 두 번째 모델의 mse ]


# 위의 출력값과 연결
print("model.metrics_names : ", model.metrics_names)
# model.metrics_names :  ['loss', 'dense_12_loss', 'dense_17_loss', 'dense_12_mae', 'dense_12_mse', 'dense_17_mae', 'dense_17_mse']

y1_predict, y2_predict = model.predict([x1_test, x2_test])
print("================================")
# print("y1_predict : \n", y1_predict)        #(20,3)
# print("y2_predict : \n", y2_predict)        #(20,3)
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

# RMSE1 :  23.600023346533355
# RMSE2 :  25.28784048088756
# RMSE :  24.443931913710458

# R2_1 :  -15.750709833290813
# R2_2 :  -18.232327103362874
# R2 :  -16.991518468326845