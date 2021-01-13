# CNN으로 구성
# 2차원을 4차원으로 늘여서 하시오
# boston : Clssification Problem

# Moudule import
import numpy as np
import tensorflow as tf
from sklearn.datasets import load_iris

# Data
iris = load_iris()
data = iris.data
feature = iris.target
print(data.shape,feature.shape) # (150, 4) (150,)

# train_test_split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(data,feature,test_size = 0.3)
print(x_train.shape,x_test.shape)   # (105, 4) (45, 4)
print(y_train.shape,y_test.shape)   # (105,) (45,)

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(np.min(x_train),np.max(x_train))  # 0.1 7.9
print(np.min(x_test),np.max(x_test))    # 0.1 7.7

x_train = x_train[...,tf.newaxis,tf.newaxis]
x_test = x_test[...,tf.newaxis,tf.newaxis]
print(x_train.shape,x_test.shape)   # (105, 4, 1, 1) (45, 4, 1, 1)

# Modeling
from tensorflow.keras.layers import Dense,Activation,Conv2D,Flatten,Dropout,Input
from tensorflow.keras.models import Model

input1 = Input(shape =(4,1,1))
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

model = Model(inputs = input1,outputs = net,name = 'iris_CNN')

# compile
model.compile(loss = 'categorical_crossentropy',optimizer = 'adam',metrics = ['acc'])

# fit
model.fit(x_train,y_train,epochs=500,validation_split = 0.2)

# evaluate
loss,acc = model.evaluate(x_test,y_test)
print('loss: ',loss)
print('acc: ',acc)

# prediction
y_pred = model.predict(x_test)

# Dense Model
# loss:  0.26552248001098633
# accuracy:  0.9111111164093018

# CNN model
# loss:  0.05330286920070648
# acc:  0.9777777791023254