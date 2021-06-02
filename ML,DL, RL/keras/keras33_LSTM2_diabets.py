# Regression Model

import numpy as np
from sklearn.datasets import load_diabetes
from tensorflow.keras import callbacks

dataset = load_diabetes()
data  = dataset.data
target = dataset.target

# print(data[:5])
# print(target[:10])
print(data.shape,target.shape)  # (442, 10) (442,)

print(np.max(data),np.min(target))
print(dataset.feature_names)
print(dataset.DESCR)

# data = data / 25.0
# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# scaler.fit(data)
# data = scaler.transform(data)

# reshape
data = data.reshape(442,10,1)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(data,target,train_size = 0.8,random_state = 121)
# x_train,x_val, y_train,y_val = train_test_split(x_train,y_train,test_size = 0.2,random_state =121)

# scaler = MinMaxScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)
# x_val = scaler.transform(x_val)

from keras.layers import Dense,Input,Activation,LSTM
from keras.models import Model

input1 = Input(shape = (10,1))
dense1 = LSTM(648)(input1)
dense1 = Activation('relu')(dense1)
dense1 = Dense(344)(dense1)
dense1 = Activation('relu')(dense1)
dense1 = Dense(128)(dense1)
dense1 = Activation('relu')(dense1)
dense1 = Dense(1)(dense1)
model = Model(inputs = input1, outputs = dense1, name = 'diabets_model')

model.compile(loss = 'mse',optimizer = 'adam',metrics = ['mae'])

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss',patience=20,mode = 'auto')

model.fit(x_train,y_train,epochs = 2000, batch_size = 8,validation_split = 0.2,callbacks = [early_stopping])

loss = model.evaluate(x_test,y_test,batch_size = 8)
y_pred = model.predict(x_test)

from sklearn.metrics import mean_squared_error
def RMSE(y_test,y_pred):
    return np.sqrt(mean_squared_error(y_test,y_pred))

print("RMSE: ", RMSE(y_test,y_pred))

from sklearn.metrics import r2_score
R2 = r2_score(y_test,y_pred)
print('R2: ', R2)

# LSTM
# RMSE:  81.61207217289866
# R2:  -0.0059410802543851116