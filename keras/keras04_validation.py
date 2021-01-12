# validation_data
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Data 
x_train = np.array([1,2,3,4,5])
y_train = np.array([1,2,3,4,5])

# x_train 이외의 컴퓨터가 데이터를 검증할 데이터를 추가한다. 훈련하면서 검증한다. > 성능 좋아짐
x_validation = np.array([6,7,8])  
y_validation = np.array([6,7,8])

x_test = np.array([9,10,11])
y_test = np.array([9,10,11])

# Model
model = Sequential([
    Dense(50000,input_dim=1, activation='relu'),
    Dense(1000),
    Dense(500),
    Dense(250),
    Dense(120),
    Dense(60),
    Dense(10),
    Dense(10),
    Dense(1)    # output : 1
])

# Compile
# model.compile(loss='mse', optimizer='adam', metrics=['accuracy']) # 출력결과 : accuracy == 0.0 (분류, 완전히 구분되어 있을 때 사용)
# model.compile(loss='mse', optimizer='adam', metrics=['mse']) # 출력결과 : mse == loss
model.compile(loss='mse', optimizer='adam', metrics=['mae']) # mae : 평균 절대 오차
# Fit
model.fit(x_train, y_train, epochs=100, batch_size=1,
          validation_data=(x_validation, y_validation)) # 검증 데이터

# Evaluate
loss = model.evaluate(x_test, y_test, batch_size=1)
print('loss : ', loss)
# Prediction
# result = model.predict([9])
result = model.predict(x_train)
print("result : ", result)