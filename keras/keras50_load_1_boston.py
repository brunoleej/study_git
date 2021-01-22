import numpy as np

data = np.load('../data/boston_data.npy')
target = np.load('../data/boston_target.npy')

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data, target, train_size = 0.8, shuffle = True, random_state=66)

# Preprocessing
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)    
x_test = scaler.transform(x_test)

print(np.min(data), np.max(data)) # 0.0 711.0      ----> 최댓값 1.0 , 최솟값 0.0
print(np.max(data[0]))  # 396.9

#2 Modeling
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(128,activation = 'relu',input_dim = 13),
    # model.add(Dense(10, activation='relu',input_shape=(13,))
    Dense(128),
    Dense(64),
    Dense(64),
    Dense(32),
    Dense(32),
    Dense(16),
    Dense(16),
    Dense(1),
])

# Compile
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

# Early Stopping
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=20, mode='auto') 

# fit
model.fit(x_train, y_train, epochs=2000, batch_size=8, validation_split=0.2, verbose=1, callbacks=[early_stopping])

#4. Evaluate
loss, mae = model.evaluate(x_test, y_test, batch_size=8)
print("loss : ", loss)
print("mae : ", mae)

# prediction
y_predict = model.predict(x_test)
# print("y_pred : \n", y_predict)

# RMSE
from sklearn.metrics import mean_squared_error
def RMSE (y_test, y_train) :
    return np.sqrt(mean_squared_error(y_test, y_train))
print("RMSE : ", RMSE(y_test, y_predict))

# R2
from sklearn.metrics import r2_score
R2 = r2_score(y_test, y_predict)
print("R2 : ", R2) 

# loss :  7.806453704833984
# mae :  2.0674474239349365
# RMSE :  2.794003868077478
# R2 :  0.9066022580201631