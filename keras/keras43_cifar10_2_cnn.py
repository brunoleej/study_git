import numpy as np
import tensorflow as tf
from keras.datasets import cifar10

(x_train,y_train),(x_test,y_test) = cifar10.load_data()
print(x_train.shape,x_test.shape)   # (50000, 32, 32, 3) (10000, 32, 32, 3)
print(np.min(x_train),np.max(x_test))   #  0 255

x_train,x_test = x_train / 255.0, x_test / 255.0
print(np.min(x_train),np.max(x_test))   # 0.0 1.0

# to_categorical
from keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Modeling
from keras.layers import Dense,Conv2D,Dropout,Flatten,MaxPool2D,Input,Activation
from keras.models import Model

input1 = Input(shape = (32,32,3))
net = Conv2D(32,3,padding='SAME')(input1)
net = Activation('relu')(net)
net = Conv2D(32,3,3,padding = 'SAME')(net)
net = Activation('relu')(net)
net = MaxPool2D(pool_size=(2,2))(net)
net = Dropout(0.25)(net)

net = Conv2D(64,3,padding='SAME')(net)
net = Activation('relu')(net)
net = Conv2D(64,3,3,padding = 'SAME')(net)
net = Activation('relu')(net)
net = MaxPool2D(pool_size=(2,2))(net)
net = Dropout(0.25)(net)

net = Flatten()(net)
net = Dense(512)(net)
net = Activation('relu')(net)
net = Dropout(0.5)(net)
net = Dense(10)(net)
net = Activation('softmax')(net)

model = Model(inputs = input1, outputs = net, name = 'sifar10_CNN_Model')

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