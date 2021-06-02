# LSTM Modeling
# binary classification
import numpy as np
from sklearn.datasets import load_breast_cancer
from tensorflow.python.keras.layers.recurrent import LSTM

# data
cancer = load_breast_cancer() 

# print(cancer.DESCR)
# print(cancer.feature_names)

x = cancer.data
y = cancer.target
print(x.shape,y.shape)  # (569,30) (569, )
print(x[:5])
print(y)

# reshape
x = x.reshape(569,30,1)

# train_test_split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,train_size = 0.7,random_state = 1)
# x_train,x_val,y_train,y_val = train_test_split(x_train,y_train,test_size = 0.3,random_state=1)

# # 전처리 / minmax, train_test_split
# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# scaler.fit(x_train)
# x_train = np.transpose(x_train)
# x_test = np.transpose(x_test)
# x_val = np.transpose(x_val)

# modeling
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense,Activation,Input,LSTM

input1 = Input(shape = (30,1))
dense1 = LSTM(300)(input1)
dense1 = Activation('sigmoid')(dense1)
dense1 = Dense(150)(dense1)
dense1 = Activation('sigmoid')(dense1)
dense1 = Dense(60)(dense1)
dense1 = Activation('sigmoid')(dense1)
dense1 = Dense(1)(dense1)
model = Model(inputs= input1, outputs = dense1)

# EarlyStopping
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='acc',patience=20,mode='auto')

# compile, fit
    # mean_squared_error
model.compile(loss = 'binary_crossentropy',optimizer='adam',metrics=['acc'])
model.fit(x_train,y_train,epochs = 500,validation_split = 0.2,callbacks = [early_stopping])

loss,acc = model.evaluate(x_test,y_test)
print('binary_crossentropy:',loss)
print('acc: ', acc)

# 실습 1. acc.0.985 이상 올릴것
# 실습 2. predict 출력해볼것
y_pred = model.predict(x[-5:-1])
print(y[-5:-1])

# # accuracy_score
# from sklearn.metrics import accuracy_score
# print('accuracy: ',accuracy_score(y_test,y_pred))

# LSTM
# binary_crossentropy: 9.742072105407715
# acc:  0.3684210479259491