import numpy as np
import tensorflow as tf
from keras.datasets import cifar10

(x_train,y_train),(x_test,y_test) = cifar10.load_data()
print(x_train.shape,x_test.shape)   # (50000, 32, 32, 3) (10000, 32, 32, 3)
print(np.min(x_train),np.max(x_test))   #  0 255
x_train = x_train.reshape(50000,3072,1)
x_test = x_test.reshape(10000,3072,1)

x_train,x_test = x_train / 255.0, x_test / 255.0
print(np.min(x_train),np.max(x_test))   # 0.0 1.0

# to_categorical
from keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Modeling
from keras.layers import Dense,Conv2D,Dropout,Flatten,MaxPool2D,Input,Activation
from keras.models import Model

input1 = Input(shape =(3072,1))
LSTM = Dense(500)(input1)
dense = Dense(500)(dense)
dense = Activation('relu')(dense)
dense = Dropout(0.25)(dense)
dense = Dense(250)(dense)
dense = Dense(250)(dense)
dense = Activation('relu')(dense)
dense = Dropout(0.25)(dense)
dense = Dense(120)(dense)
dense = Dense(120)(dense)
dense = Activation('relu')(dense)
dense = Dropout(0.25)(dense)
dense = Dense(60)(dense)
dense = Dense(60)(dense)
dense = Activation('relu')(dense)
dense = Dropout(0.25)(dense)
dense = Dense(30)(dense)
dense = Dense(30)(dense)
dense = Activation('relu')(dense)
dense = Dropout(0.25)(dense)
dense = Dense(10)(dense)
dense = Activation('softmax')(dense)

model = Model(inputs = input1, outputs = dense, name = 'sifar10_DNN_Model')

# Compile
model.compile(loss = 'categorical_crossentropy',metrics = ['acc'])

# fit
model.fit(x_train,y_train,batch_size = 64,shuffle=True,epochs = 5)

# Evaluate
loss,acc = model.evaluate(x_test,y_test,batch_size = 64)
print('loss : ',loss)
print('accuracy: ',acc)

# CNN
# loss :  1.1537483930587769
# accuracy:  0.585099995136261

# DNN
# loss :  2.3026063442230225
# accuracy:  0.10000000149011612