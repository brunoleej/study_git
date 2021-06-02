# RMSE
# y_predict와 y_test간의 RMSE를 통해서 얼마나 정확한 예측을 했는지 평가
# sklearn 사용
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import numpy as np

# Data
x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([1,2,3,4,5,6,7,8,9,10])

x_test = array([11,12,13,14,15])
y_test = array([11,12,13,14,15])

x_pred = array([16,17,18])

# Modeling
model = Sequential()
model.add(Dense(10, input_dim=1, activation='relu'))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(1))

# Compile
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
# Fit
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.2) 

# Evaluate
results = model.evaluate(x_test, y_test, batch_size=1) #loss = 'mse', metrics='mae' 값이 들어간다
print("mse, mae :", results)
# Prediction
y_predict = model.predict(x_test)
# print("y_pred: ", y_predict)

# Definition RMSE function
# RMSE : 낮아야 좋다.
from sklearn.metrics import mean_squared_error #mse
def RMSE (y_test, y_predict) :
    return np.sqrt(mean_squared_error(y_test, y_predict)) #RMSE = mse에 루트를 씌운다.
print("RMSE :", RMSE(y_test, y_predict))

# print("mse :", mean_squared_error(y_test, y_predict))
print("mse :", mean_squared_error(y_predict,y_test))

# 실습 : RMSE 를 0.1 이하로 낮추시오