import numpy as np
from sklearn.datasets import load_diabetes

dataset = load_diabetes()

# Data
data = dataset.data
target = dataset.target

print(data[:5])
print(target[:10])

print(x.shape, y.shape)         #(442, 10) (442,) input = 10, output = 1
print(np.max(data), np.min(target))     # 0.198787989657293 25.0  ---> 전처리 해야 함
print(np.max(data), np.min(data))     # 0.198787989657293 -0.137767225690012
print(dataset.feature_names)    # 10 column
                                # ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
# print(dataset.DESCR)

# train_test_split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data, target, train_size=0.8, shuffle=True)

# Preprocessing
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)     
x_test = scaler.transform(x_test)


#2. Modeling
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(160,activation = 'linear',input_dim = 10),
    Dense(160),
    Dense(80,activation = 'linear'),
    Dense(80),
    Dense(40,activation = 'linear'),
    Dense(40),
    Dense(20,activation = 'linear'),
    Dense(20),
    Dense(1)
])

#3. Compile
model.compile(loss='mse', optimizer='adam',metrics=['mae'] )

# EarlyStopping
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=20, mode='min') 

# fit
model.fit(x_train, y_train, epochs=5000, batch_size=5, validation_split=0.2, verbose=1,callbacks=[early_stopping] )

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

# Early Stopping
# loss :  2669.86962890625
# mae :  42.07997512817383
# RMSE :  51.67078489268979
# R2 :  0.5338410030424983
