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
print(x)
print(x.shape)

# Model
from tensorflow.keras.layers import Dense,GRU
from tensorflow.keras.models import Sequential
model = Sequential([
    GRU(10,activation = 'relu',input_shape=(3,1)),
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

# LSTM
# loss:  0.05855182558298111
# y_pred:  [[80.57456]]

# SimpleRNN
# loss:  0.02435765601694584
# y_pred:  [[80.52275]]

# GRU
# loss:  0.3453718423843384
# y_pred:  [[80.98243]]