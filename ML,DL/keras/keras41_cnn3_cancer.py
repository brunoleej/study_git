# CNN으로 구성
# 2차원을 4차원으로 늘여서 하시오
# boston : Clssification Problem
import numpy as np
import tensorflow as tf
from sklearn.datasets import load_breast_cancer

# Data
cancer = load_breast_cancer()
data = cancer.data
feature = cancer.target

# train_test_split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(data,feature,test_size = 0.3,shuffle = False)
print(x_train.shape,x_test.shape)   # (398, 30) (171, 30)
print(np.min(x_train),np.max(x_train))  # 0.0 3432.0
print(np.min(x_test),np.max(x_test))    # 0.0 4254.0

# preprocessing
from sklearn.preprocessing import MinMaxScaler
scaler1 = MinMaxScaler()
scaler2 = MinMaxScaler()
scaler1.fit(x_train)
scaler2.fit(x_test)
x_train = scaler1.transform(x_train)
x_test = scaler2.transform(x_test)
x_test = x_test / 1.0000000000000002

x_train = x_train[...,tf.newaxis,tf.newaxis]
x_test = x_test[...,tf.newaxis,tf.newaxis]
print(x_train.shape,x_test.shape)   # (398, 30, 1, 1) (171, 30, 1, 1)
print(np.min(x_train),np.max(x_train))  # 0.0 1.0
print(np.min(x_test),np.max(x_test))    # 0.0 1.0

# Modeling
from tensorflow.keras.layers import Dense,Conv2D,Flatten,Activation,Dropout,Input
from tensorflow.keras.models import Model

input1 = Input(shape = (30,1,1))
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
net = Activation('sigmoid')(net)

model = Model(inputs = input1,outputs = net,name = 'breast_cancer_CNN')

# Compile
model.compile(loss = 'binary_crossentropy',optimizer='adam',metrics=['accuracy'])

# fit
model.fit(x_train,y_train,epochs=500,validation_split=0.2,batch_size=3)

# evaluate
loss, accuracy = model.evaluate(x_test,y_test,batch_size = 3)
print('loss: ',loss)
print('accuracy: ', accuracy)

# predict
y_pred = model.predict(x_test)

# Dense Model
# loss:  0.2283373773097992
# accuracy:  0.9181286692619324

# CNN Model
# loss:  0.22723810374736786
# accuracy:  0.9707602262496948

