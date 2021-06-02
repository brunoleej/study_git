# Classification Problem
import numpy as np
from sklearn.datasets import load_breast_cancer

# Data
cancer = load_breast_cancer() 

print(cancer.DESCR)
print(cancer.feature_names)

data = cancer.data
target = cancer.target
print(data.shape)  # (569,30)
print(target.shape)  # (569, )
print(x[:5])
print(y)

# train_test_split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(data,target,train_size = 0.7,random_state = 1)

# 전처리 / minmax, train_test_split
# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# scaler.fit(x_train)
# x_train = np.transpose(x_train)
# x_test = np.transpose(x_test)
# x_val = np.transpose(x_val)

# modeling
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense,Activation,Input

input1 = Input(shape = (30,))
dense1 = Dense(300)(input1)
dense1 = Dense(300)(dense1)
dense1 = Activation('relu')(dense1)
dense1 = Dense(150)(dense1)
dense1 = Dense(150)(dense1)
dense1 = Activation('relu')(dense1)
dense1 = Dense(60)(dense1)
dense1 = Dense(60)(dense1)
dense1 = Activation('relu')(dense1)
dense1 = Dense(30)(dense1)
dense1 = Dense(30)(dense1)
dense1 = Activation('relu')(dense1)
dense1 = Dense(1)(dense1)
dense1 = Activation('sigmoid')(dense1)
model = Model(inputs= input1, outputs = dense1)

# compile
model.compile(loss = 'binary_crossentropy',optimizer='adam',metrics=['acc'])

# EarlyStopping
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor = 'acc',patience = 20,mode = 'auto')

# fit
model.fit(x_train,y_train,epochs = 300,validation_split = 0.2,callbacks=[early_stopping])

# Evaluate
loss,acc = model.evaluate(x_test,y_test)
print('loss: ',loss)
print('accuracy: ',acc)

# 실습 1. acc.0.985 이상 올릴것
# 실습 2. predict 출력해볼것
y_pred = model.predict(data[-5:-1])
print(target[-5:-1])

# loss:  0.2283373773097992
# accuracy:  0.9181286692619324