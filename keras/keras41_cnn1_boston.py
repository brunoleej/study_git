# CNN으로 구성
# 2차원을 4차원으로 늘여서 하시오
# boston : Regression Problem

# Moudule import 
import numpy as np
import tensorflow as tf
from sklearn.datasets import load_boston

# Data
boston = load_boston()
data = boston.data
feature = boston.target
print(data.shape,feature.shape) # (506, 13) (506,) 
# print(np.min(data),np.max(data))    # 0.0 711.0
# print(np.min(feature),np.max(feature))  # 5.0 50.0


# train_test_split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(data,feature,test_size = 0.3,shuffle = True)
print(x_train.shape,y_train.shape)  # (354, 13) (354,)
print(x_test.shape,y_test.shape)    # (152, 13) (152,)
print(np.min(x_train),np.max(x_train))  # 0.0 711.0
print(np.min(x_test),np.max(x_test))    # 0.0 711.0
# print(x_train.dtype)    # float64

# channel 추가
x_train = x_train[...,tf.newaxis,tf.newaxis]
x_test = x_test[...,tf.newaxis,tf.newaxis]
print(x_train.shape,x_test.shape)   # (354, 13, 1,1) (152, 13, 1,1)

# Data Normalization
x_train,x_test = x_train / 711.0, x_test / 711.0
print(np.min(x_train),np.max(x_train))  
print(np.min(x_test),np.max(x_test))

# Modeling
from tensorflow.keras.layers import Conv2D,Dense,Activation,Input,Dropout,Flatten
from tensorflow.keras.models import Model

# Fully Connected
input1 = Input(shape=(13,1,1))
net = Conv2D(32,3,3,padding='SAME')(input1)
net = Activation('relu')(net)
net = Conv2D(32,3,3,padding='SAME')(net)
net = Activation('relu')(net)
net = Dropout(0.25)(net)

net = Conv2D(64,3,3,padding='SAME')(input1)
net = Activation('relu')(net)
net = Conv2D(64,3,3,padding='SAME')(net)
net = Activation('relu')(net)
net = Dropout(0.25)(net)

net = Flatten()(net)
net = Dense(512)(net)
net = Activation('relu')(net)
net = Dropout(0.5)(net)
net = Dense(1)(net)
net = Activation('linear')(net)

model = Model(inputs = input1,outputs = net,name = 'Boston_housing_CNN')

# compile
model.compile(loss = 'mse',optimizer = 'adam',metrics = ['mae']) 

# fit
model.fit(x_train,y_train,epochs=500,validation_split=0.2,batch_size = 5)

# evaluate
loss,mae = model.evaluate(x_test,y_test,batch_size=5)
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

# early_stopping (5)
# loss :  10.76313304901123
# mae :  2.4629220962524414
# RMSE :  3.2807214371463127
# R2 :  0.8712281059778333

# early_stopping (10) 
# loss :  8.8392915725708
# mae :  2.440977096557617
# RMSE :  2.973094694598015
# R2 :  0.8942452569241743

# early_stopping (20) 
# loss :  6.976583957672119
# mae :  2.0358965396881104
# RMSE :  2.6413223452327177
# R2 :  0.91653100556001

# CNN model
# loss:  21.7302303314209
# mse:  3.436227560043335
# RMSE:  4.6615693080766
# R2:  0.7145729875723015

# CNN model final output Activation 'linear'
# loss:  21.139278411865234
# mse:  3.452582836151123
# RMSE:  4.597746488205329
# R2:  0.7020289867320698

# CNN model final output Activation 'linear'(second try)
# loss:  21.682680130004883
# mse:  3.6008100509643555
# RMSE:  4.6564665050609255
# R2:  0.7518755394965633