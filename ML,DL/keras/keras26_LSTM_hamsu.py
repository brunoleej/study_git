# keras23_LSTM3_scale을 함수형으로 코딩
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
from keras.layers import Dense,LSTM,Input,Activation
from tensorflow.keras.models import Model
input1 = Input(shape=(3,1))
lstm = LSTM(10)(input1)
lstm = Activation('relu')(lstm)
dense1 = Dense(30)(lstm)
dense1 = Dense(20)(dense1)
dense1 = Dense(10)(dense1)
dense1 = Activation('relu')(dense1)
dense1 = Dense(5)(dense1)
dense1 = Dense(1)(dense1)
model = Model(inputs = input1, outputs = dense1,name = 'Functional_LSTM')


# print(model.summary())

# Earlystopping
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor = 'loss',patience=30,mode = 'auto')

# Compile, Train
model.compile(loss = 'mse',optimizer = 'adam')
model.fit(x,y,epochs = 1000,batch_size = 1)

# x_pred.reshape
x_pred = x_pred.reshape(1,3,1).astype('float32')
print(x_pred.shape)

# evaluate, predict
loss = model.evaluate(x,y)
print('loss: ',loss)

y_pred = model.predict(x_pred)
print('y_pred: ',y_pred)

# loss:  0.05855182558298111
# y_pred:  [[80.57456]]