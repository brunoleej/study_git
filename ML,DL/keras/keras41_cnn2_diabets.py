# CNN으로 구성
# 2차원을 4차원으로 늘여서 하시오
# boston : Regression Problem
import numpy as np
import tensorflow as tf
from sklearn.datasets import load_diabetes

# Data
diabetes = load_diabetes()
data = diabetes.data
feature = diabetes.target
print(data.shape,feature.shape) # (442, 10) (442,)

# train_test_split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(data,feature,test_size = 0.2,shuffle = True)
print(x_train.shape,x_test.shape)   # (353, 10) (89, 10)

# MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
scaler1 = MinMaxScaler()
scaler2 = MinMaxScaler()
sc1 = scaler1.fit(x_train)
sc2 = scaler2.fit(x_test)
x_train = scaler1.transform(x_train)
x_test = scaler2.transform(x_test)
print(np.min(x_train),np.max(x_train))  # 0.0 1.0
print(np.min(x_test),np.max(x_test))    # 0.0 1.0000000000000002

# Normalization
x_train = x_train[...,tf.newaxis,tf.newaxis]
x_test = x_test[...,tf.newaxis,tf.newaxis]
print(x_train.shape,x_test.shape)   # (353, 10, 1, 1) (89, 10, 1, 1)

print(np.min(x_train),np.max(x_train))  # 0.0 1.0
print(np.min(x_test),np.max(x_test))   # 0.0 1.0000000000000002

x_test = x_test / 1.0000000000000002
print(np.max(x_test))   # 1.0

# Modeling
from tensorflow.keras.layers import Dense,Conv2D,Flatten,Input,Dropout,Activation
from tensorflow.keras.models import Model

input1 = Input(shape=(10,1,1))
net = Conv2D(32,3,3,padding='SAME')(input1)
net = Activation('relu')(net)
net = Conv2D(32,3,3,padding='SAME')(net)
net = Activation('relu')(net)
net = Dropout(0.25)(net)

net = Conv2D(64,3,3,padding='SAME')(net)
net = Activation('relu')(net)
net = Conv2D(64,3,3,padding='SAME')(net)
net = Activation('relu')(net)
net = Dropout(0.25)(net)

net = Flatten()(net)
net = Dense(512)(net)
net = Activation('relu')(net)
net = Dropout(0.5)(net)
net = Dense(1)(net)

model = Model(inputs = input1,outputs = net,name = 'diabetes_CNN')

# compile
model.compile(loss = 'mse',optimizer = 'adam',metrics = ['mae']) 

# fit
model.fit(x_train,y_train,epochs=500,validation_split=0.2,batch_size = 1)

# evaluate
loss,mae = model.evaluate(x_test,y_test,batch_size=1)
print('loss: ', loss)
print('mse: ',mae)

# Prediction
y_pred = model.predict(x_test)

# RMSE
from sklearn.metrics import mean_squared_error
def RMSE(y_test,y_pred):
    return np.sqrt(mean_squared_error(y_test,y_pred))
print('RMSE: ',RMSE(y_test,y_pred))

# R2
from sklearn.metrics import r2_score
def R2(y_test,y_pred):
    return r2_score(y_test,y_pred)
print('R2: ', R2(y_test,y_pred))

# x_train만 preprocessing
# loss :  3136.124755859375
# mae :  46.16136932373047
# RMSE :  56.00111389680871
# R2 :  0.5167788542278127

# CNN diabets
# loss:  5271.126953125
# mse:  57.096527099609375
# RMSE:  72.60252481119431
# R2:  0.07872804961406843