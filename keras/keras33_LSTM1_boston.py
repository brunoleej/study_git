# sklearn Dataset use
# LSTM modeling
# Dense와 성능비교
# Regression model

import numpy as np

# Data
from sklearn.datasets import load_boston
boston = load_boston()
x = boston.data
y = boston.target
print(x.shape,y.shape)  # (506, 13) (506,)

# reshape for LSTM
x = x.reshape(506,13,1)
print(x.shape)  # (506, 13, 1) 

# train_test_split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state = 121)

# # Preprocessing
# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# scaler.fit(x_train)
# x_train = np.transform(x_train)
# x_test = np.transfrom(x_test)

# Model
from keras.layers import Dense,LSTM,Input,Activation
from keras.models import Model

input1 = Input(shape = (13,1))
dense1 = LSTM(300)(input1)
dense1 = Dense(150)(dense1)
dense1 = Dense(74)(dense1)
dense1 = Dense(36)(dense1)
dense1 = Dense(1)(dense1)
model = Model(inputs=input1,outputs = dense1)

# EarlyStopping
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor = 'mae',patience=20,mode = 'auto')

# compile,fit
model.compile(loss = 'mse',optimizer='adam',metrics=['mae'])
model.fit(x_train,y_train,epochs=1000,validation_split = 0.2,callbacks=[early_stopping])

# loss
loss,mae = model.evaluate(x_test,y_test)
print('loss: ',loss)
print('mae: ', mae)

# predcit
y_pred = model.predict(x_test)

from sklearn.metrics import mean_squared_error
def RMSE(y_test,y_pred):
    return np.sqrt(mean_squared_error(y_test,y_pred))

print("RMSE: ", RMSE(y_test,y_pred))

from sklearn.metrics import r2_score
R2 = r2_score(y_test,y_pred)
print('R2: ', R2)

# LSTM
# loss:  22.448734283447266
# mae:  3.2822868824005127
# RMSE:  4.7380094888671
# R2:  0.7243016887865155
