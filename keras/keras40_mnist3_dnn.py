# 주말 과제
# Dense 모델로 구성한 input_shape = (28*28,)
# (N,784)

# Data
import numpy as np
import tensorflow as tf
from keras.datasets import mnist

# preprocessing
(x_train,y_train),(x_test,y_test) = mnist.load_data()
print(x_train.shape,y_train.shape)    # (60000, 28, 28) (60000,)
print(x_test.shape,y_test.shape)    # (10000, 28, 28) (10000,)
print(np.min(x_train),np.max(x_train))  # 0 255
print(np.min(x_test),np.max(x_test))    # 0 255

x_train,x_test = x_train / 255.0, x_test / 255.0
print(np.min(x_train),np.max(x_train))  # 0.0 1.0
print(np.min(x_test),np.max(x_test))    # 0.0 1.0

x_train = x_train.reshape(60000,784)    
x_test = x_test.reshape(10000, 784)
print(x_train.shape,x_test.shape)   # (60000, 784) (10000, 784)

# to_categorical
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# modeling
from tensorflow.keras.layers import Dense,Activation,Input
from tensorflow.keras.models import Model
input1 = Input(shape = (784,))
dense = Dense(500)(input1)
dense = Dense(500)(dense)
dense = Activation('relu')(dense)
dense = Dense(250)(dense)
dense = Dense(250)(dense)
dense = Activation('relu')(dense)
dense = Dense(120)(dense)
dense = Dense(120)(dense)
dense = Activation('relu')(dense)
dense = Dense(60)(dense)
dense = Dense(60)(dense)
dense = Activation('relu')(dense)
dense = Dense(30)(dense)
dense = Dense(30)(dense)
dense = Activation('relu')(dense)
dense = Dense(10)(dense)
dense = Activation('softmax')(dense)

model = Model(inputs = input1,outputs = dense,name = 'mnist_Dense')

# compile
model.compile(loss = 'categorical_crossentropy',optimizer = 'adam',metrics=['acc'])

# fit
model.fit(x_train,y_train,epochs = 5,batch_size=64)

# evlauate
loss,acc = model.evaluate(x_test,y_test,batch_size = 64)
print('loss: ',loss)
print('acc: ',acc)

# CNN Model
# loss :  0.04032869264483452
# accuracy : 0.987500011920929

# DNN Model 
# loss:  0.12349266558885574
# acc:  0.9685999751091003