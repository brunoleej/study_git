# 당뇨병 회귀모델
# 실습 : 18에서 했던 것과 동일하게 19-1,2,3,4,5, EarlyStopping 까지 총 6개의 파일을 완성하시오.
# [2] sklearn >> MinMaxScaler (전체 x 전처리)

import numpy as np
from sklearn.datasets import load_diabetes

# Data
diabetes = load_diabetes()
x = diabetes.data
y = diabetes.target

print(x[:5])
print(y[:10])

print(x.shape, y.shape)         #(442, 10) (442,) input = 10, output = 1
print(np.max(x), np.min(y))     # 0.198787989657293 25.0  ---> 전처리 해야 함
print(np.max(x), np.min(x))     # 0.198787989657293 -0.137767225690012
print(diabetes.feature_names)    # 10 column
                                # ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
# print(diabetes.DESCR)

# 전처리 과정
# x = x/0.198787989657293

# Preprocessing
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x)     
print(np.max(x), np.min(x)) # 최댓값 1.0 , 최솟값 0.0
print(np.max(x[0]))         # max = 1   -----> 컬럼마다 최솟값과 최댓값을 적용해서 구해준다.

# train_test_split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

print(x_train.shape)    #(353, 10)
print(x_test.shape)     #(89, 10)


#2. Modeling
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(500,activation = 'linear',input_dim = 10),
    Dense(500),
    Dense(250),
    Dense(250),
    Dense(120),
    Dense(120),
    Dense(60),
    Dense(30),
    Dense(10),
    Dense(1)
])

#3. Compile, fit
model.compile(loss='mse', optimizer='adam',metrics=['mae'] )
model.fit(x_train, y_train, epochs=200, batch_size=5, validation_split=0.2, verbose=1)

#4. Evaluate
loss, mae = model.evaluate(x_test, y_test, batch_size=5)
print("loss : ", loss)
print("mae : ", mae)

# Prediction
y_predict = model.predict(x_test)

# RMSE
from sklearn.metrics import mean_squared_error
def RMSE (y_test, y_predict) :
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))

# R2
from sklearn.metrics import r2_score
r2 = r2_score (y_test, y_predict)
print("R2 : ", r2)

# MinMaxscler 전처리 후 
# loss :  3275.446044921875
# mae :  46.802120208740234
# RMSE :  57.231511897763696
# R2 :  0.49531193139582375
