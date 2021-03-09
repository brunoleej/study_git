# input_shape / input_length / input_dim
import numpy as np

# 1. Data
x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6]])
y = np.array([4,5,6,7])

print(x.shape,y.shape)  # (4, 3) (4,)

x = x.reshape(4,3,1)

# model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

model = Sequential([
    # LSTM(10,activation = 'relu',input_shape = (3,1)),
    LSTM(10,activation = 'relu',input_length = 3,input_dim = 1),
    Dense(20),
    Dense(10),
    Dense(1)
])
# print(model.summary())

#3. compile, train
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x,y,epochs = 100, batch_size = 1)

# evaluate, predict
loss = model.evaluate(x,y)
print(loss)

x_pred = np.array([5,6,7]) # (3,) -> (1,3,1)
x_pred = x_pred.reshape(1,3,1)

result = model.predict(x_pred)
print(result)
