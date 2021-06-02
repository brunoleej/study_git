# 당뇨병 회귀모델
# 실습 : 18에서 했던 것과 동일하게 19-1,2,3,4,5, EarlyStopping 까지 총 6개의 파일을 완성하시오.
# [4] x_train 데이터만 전처리 한다.
# validation data
import numpy as np
from sklearn.datasets import load_diabetes

# Data
dibetes = load_diabetes()
x = dibetes.data
y = dibetes.target

print(x[:5])
print(y[:10])

print(x.shape, y.shape)         #(442, 10) (442,) input = 10, output = 1

print(np.max(x), np.min(y))     # 0.198787989657293 25.0  ---> 전처리 해야 함
print(np.max(x), np.min(x))     # 0.198787989657293 -0.137767225690012
print(dibetes.feature_names)    # 10 column
                                # ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
# print(dibetes.DESCR)

# 전처리 과정
# x = x/0.198787989657293

# train_test_split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)
x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, train_size=0.8, shuffle=True, random_state=66)

# Preprocessing
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)     
x_test = scaler.transform(x_test)
x_validation = scaler.transform(x_validation)

#2. Modeling
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
model = Sequential([
    Dense(300,activation = 'linear',input_dim = 10),
    Dense(300),
    Dense(150,activation = 'linear'),
    Dense(150),
    Dense(70,activation = 'linear'),
    Dense(70),
    Dense(30,activation = 'linear'),
    Dense(30),
    Dense(1)
])

# Compile, fit
model.compile(loss='mse', optimizer='adam',metrics=['mae'] )
model.fit(x_train, y_train, epochs=100, batch_size=5, validation_data=(x_validation, y_validation), verbose=1)

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

# x_train 만 MinMaxScaler 전처리 후
# loss :  3136.124755859375
# mae :  46.16136932373047
# RMSE :  56.00111389680871
# R2 :  0.5167788542278127