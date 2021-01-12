# keras23_3 copy
import numpy as np

# 1. Data
x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
              [5,6,7],[6,7,8],[7,8,9],[8,9,10],
              [9,10,11],[10,11,12],
              [20,30,40],[30,40,50],[40,50,60]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x_pred = np.array([50,60,70])

# LSTM
# result => 80
print(x.shape,y.shape) # (13, 3) (13,)
x = x.reshape((13,3,1)).astype('float32')
# x = x.reshape(x.shape[0],x.shape[1],1)
print(x)
print(x.shape)

# Model
from tensorflow.keras.layers import Dense,LSTM
from tensorflow.keras.models import Sequential
model = Sequential([
    LSTM(10,activation = 'relu',return_sequences=True,input_shape=(3,1)),
    LSTM(20),
    Dense(30),
    Dense(20),
    Dense(10,activation = 'relu'),
    Dense(5),
    Dense(1)
])
# print(model.summary())

# Earlystopping
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor = 'loss',patience=20,mode = 'auto')

# Compile, Train
model.compile(loss = 'mse',optimizer = 'adam')
model.fit(x,y,epochs = 500,batch_size = 1,callbacks=[early_stopping])

# x_pred.reshape
x_pred = x_pred.reshape(1,3,1).astype('float32')
print(x_pred.shape)

# evaluate, predict
loss = model.evaluate(x,y)
print('loss: ',loss)

y_pred = model.predict(x_pred)
print('y_pred: ',y_pred)

# 1 LSTM
# loss:  0.05855182558298111
# y_pred:  [[80.57456]]

# 2 LSTM
# loss:  0.36227428913116455
# y_pred:  [[72.83236]]
