import numpy as np
import tensorflow as tf

c10_x_train = np.load('../data/c10_x_train.npy')
c10_y_train = np.load('../data/c10_y_train.npy')
c10_x_test = np.load('../data/c10_x_test.npy')
c10_y_test = np.load('../data/c10_y_test.npy')
print(c10_x_train.shape,c10_x_test.shape)   # (50000, 32, 32, 3) (10000, 32, 32, 3)
print(np.min(c10_x_train),np.max(c10_x_test))   #  0 255

c10_x_train,c10_x_test = c10_x_train / 255.0, c10_x_test / 255.0
print(np.min(c10_x_train),np.max(c10_x_test))   # 0.0 1.0

# to_categorical
from keras.utils import to_categorical
c10_y_train = to_categorical(c10_y_train)
c10_y_test = to_categorical(c10_y_test)

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
model.fit(c10_x_train,c10_y_train,batch_size = 64,shuffle=True,epochs = 5)

# Evaluate
loss,acc = model.evaluate(c10_x_test,c10_y_test,batch_size = 64)
print('loss : ',loss)
print('accuracy: ',acc)

# loss :  1.3831406831741333
# accuracy:  0.5281000137329102