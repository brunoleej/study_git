# CNN으로 구성
# 2차원을 4차원으로 늘여서 하시오
# boston : Clssification Problem

# Moudule import 
import numpy as np
import tensorflow as tf
from sklearn.datasets import load_wine

wine = load_wine()
data = wine.data
feature = wine.target
print(data.shape,feature.shape) # (178, 13) (178,)

# preprocessing
# train_test_split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(data,feature,test_size = 0.3)
print(x_train.shape,x_test.shape)   # (124, 13) (54, 13)

print(np.min(x_train),np.max(x_train))  # 0.13 1680.0
print(np.min(x_test),np.max(x_test))    # 0.14 1547.0

# MinmaxScalar
from sklearn.preprocessing import MinMaxScaler
scaler1 = MinMaxScaler()
scaler2 = MinMaxScaler()
scaler1.fit(x_train)
scaler2.fit(x_test)
x_train = scaler1.transform(x_train)
x_test = scaler2.transform(x_test)
print(np.min(x_train),np.max(x_train))  # 0.0 1.0
print(np.min(x_test),np.max(x_test))    # 0.0 1.0000000000000004

x_test = x_test / 1.0000000000000002
print(np.min(x_test),np.max(x_test))    # 0.0 1.0

# to_categorical
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# add channel
x_train = x_train[...,tf.newaxis,tf.newaxis]
x_test = x_test[...,tf.newaxis,tf.newaxis]
print(x_train.shape,x_test.shape)   # (124, 13, 1, 1) (54, 13, 1, 1)

# Modeling
from tensorflow.keras.layers import Dense,Conv2D,Flatten,Dropout,Input,Activation
from tensorflow.keras.models import Model

input1 = Input(shape = (13,1,1))
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
net = Dense(3)(net)
net = Activation('softmax')(net)

model = Model(inputs = input1,outputs = net,name = 'wine_CNN')

# compile
model.compile(loss = 'categorical_crossentropy',optimizer = 'adam',metrics=['acc'])

# fit
model.fit(x_train,y_train,epochs = 500,validation_split=0.2,batch_size = 3)

# evaluate
loss,acc = model.evaluate(x_test,y_test,batch_size = 3)
print('loss: ',loss)
print('acc: ',acc)

# prediction
y_pred = model.predict(x_test)

# Dense Model
# loss:  1.0445780754089355
# accuracy:  0.7592592835426331

# CNN Model
# loss:  0.326945424079895
# acc:  0.9629629850387573