import numpy as np

data = np.load('../data/diabetes_data.npy')
target = np.load('../data/diabetes_target.npy')

print(data.shape, target.shape)         #(442, 10) (442,) input = 10, output = 1
print(np.min(data), np.max(target))     # 0.198787989657293 25.0  ---> 전처리 해야 함
print(np.min(data), np.max(data))     # 0.198787989657293 -0.137767225690012

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

# loss :  3484.218505859375
# mae :  46.79964065551758
# RMSE :  59.02727059875423
# R2 :  0.43294032478272326
