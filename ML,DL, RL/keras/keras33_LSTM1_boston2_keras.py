# LSTM Modeling
# LSTM modeling
# Dense와 성능비교
# Regression model
import numpy as np
from keras.datasets import boston_housing

(train_data,train_targets),(test_data,test_targets) = boston_housing.load_data()

print(np.min(train_data), np.max(train_data))

from sklearn.model_selection import train_test_split

train_data,data_val,train_targets,target_val = train_test_split(train_data,train_targets,test_size = 0.2,random_state = 121)
print(train_data.shape, train_targets.shape)    # (323, 13) (323,)

train_data = train_data.reshape(323,13,1)
print(train_data.shape) # (323, 13, 1)

# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# scaler.fit(train_data)
# train_data = scaler.transform(train_data)
# test_data = scaler.transform(test_data)
# data_val = scaler.transform(data_val)

from keras.layers import Dense,Input,Activation,LSTM
from keras.models import Model

input1 = Input(shape = (13,1))
dense1 = LSTM(328)(input1)
dense1 = Activation('relu')(dense1)
dense1 = Dense(128)(dense1)
dense1 = Activation('relu')(dense1)
dense1 = Dense(64)(dense1)
dense1 = Activation('relu')(dense1)
dense1 = Dense(1)(dense1)
model = Model(inputs = input1, outputs = dense1,name  = 'keras_boston_housing')

model.compile(loss = 'mse', optimizer = 'adam',metrics = ['mae'])
model.fit(train_data,train_targets,epochs = 100, batch_size = 8,validation_data = (data_val,target_val))

loss, mae = model.evaluate(test_data,test_targets,batch_size = 8)
print('loss: ',loss)
print('mae: ',mae)

y_pred = model.predict(test_data)

from sklearn.metrics import mean_squared_error
def RMSE(test_targets,y_pred):
    return np.sqrt(mean_squared_error(test_targets,y_pred))
print('RMSE: ',RMSE(test_targets,y_pred))

from sklearn.metrics import r2_score
def R2(test_targets,y_pred):
    return r2_score(test_targets,y_pred)
print('R2: ', R2(test_targets,y_pred))

# LSTM
# loss:  23.912973403930664
# mae:  3.4103524684906006
# RMSE:  4.8900893550158955
# R2:  0.7127358438461442