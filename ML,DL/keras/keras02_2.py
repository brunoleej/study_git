import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
# from tensorflow.keras import models
# from tensorflow import keras

from tensorflow.keras.layers import Dense


# Data 
x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([2,4,6,8,10,12,14,16,18,20])

x_test = np.array([101,102,103,104,105,106,107,108,109,110])
y_test = np.array([111,112,113,114,115,116,117,118,119,120])

x_predict = np.array([111,112,113])

# Model
model = Sequential()
# model = models.Sequential()
# model = keras.models.Sequential()

model.add(Dense(250, input_dim=1, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10))
model.add(Dense(50))
model.add(Dense(100))
model.add(Dense(1))

# Compile
model.compile(loss='mse', optimizer='adam')
# Fit
model.fit(x_train, y_train, epochs=200, batch_size=1)

# Evaluate
loss = model.evaluate(x_test, y_test, batch_size=1)
print('loss : ', loss) #loss 값을 줄여라
# Prediction
result = model.predict(x_predict)
print("result : ", result)