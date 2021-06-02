# 당뇨병 회귀모델
# 실습 : 18에서 했던 것과 동일하게 19-1,2,3,4,5, EarlyStopping 까지 총 6개의 파일을 완성하시오.
# 다 : 1 mlp 모델
# 전처리 전

import numpy as np
from sklearn.datasets import load_diabetes

# Data
diabetes = load_diabetes()
x = diabetes.data
y = diabetes.target

print(x[:5])
print(y[:10])

print(x.shape, y.shape)         #(442, 10) (442,) input = 10, output = 1
print(np.max(x), np.min(y))     #0.198787989657293 25.0  ---> 전처리 해야 함
print(diabetes.feature_names)    # 10 column
                                # ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
# print(diabetes.DESCR)

# train_test_split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)
# print(x_train.shape)    #(353, 10)
# print(x_test.shape)     #(89, 10)

#2. Modeling
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(150,activation = 'linear',input_dim = 10),
    Dense(150),
    Dense(70),
    Dense(70),
    Dense(30),
    Dense(30),
    Dense(10),
    Dense(10),
    Dense(1)
])

# Compile, fit
model.compile(loss='mse', optimizer='adam',metrics=['mae'] )
model.fit(x_train, y_train, epochs=100, batch_size=5, validation_split=0.2, verbose=1)

# Evaluate
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

# 전처리 전
# loss :  3279.437255859375
# mae :  46.741485595703125
# RMSE :  57.26637009734545
# R2 :  0.49469695987377804
