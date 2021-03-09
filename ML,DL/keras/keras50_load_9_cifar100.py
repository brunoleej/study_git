import numpy as np
import tensorflow as tf

c100_x_train = np.load('../data/c100_x_train.npy')
c100_y_train = np.load('../data/c100_y_train.npy')
c100_x_test = np.load('../data/c100_x_test.npy')
c100_y_test = np.load('../data/c100_y_test.npy')

print(c100_x_train.shape,c100_x_test.shape)   # (50000, 32, 32, 3) (10000, 32, 32, 3)
print(np.min(c100_x_train),np.max(c100_x_test))   #  0 255
print(c100_y_train.shape,c100_y_test.shape)   # (50000, 1) (10000, 1)

c100_x_train,c100_x_test = c100_x_train / 255.0, c100_x_test / 255.0
print(np.min(c100_x_train),np.max(c100_x_test))   # 0.0 1.0

# to_categorical
from keras.utils import to_categorical
c100_y_train = to_categorical(c100_y_train)
c100_y_test = to_categorical(c100_y_test)

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
net = Dense(100)(net)
net = Activation('softmax')(net)

model = Model(inputs = input1, outputs = net, name = 'sifar10_CNN_Model')

# Compile
model.compile(loss = 'categorical_crossentropy',metrics = ['acc'])

# fit
model.fit(c100_x_train,c100_y_train,batch_size = 32,shuffle=True,epochs = 10)

# Evaluate
loss,acc = model.evaluate(c100_x_test,c100_y_test,batch_size = 32)
print('loss : ',loss)
print('accuracy: ',acc)

# loss :  3.150128126144409
# accuracy:  0.23829999566078186