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

# Earlystooping, ModelCheckpoint
from keras.callbacks import EarlyStopping,ModelCheckpoint
# modelpath = './modelCheckpoint/k45_cifar10_{epoch:02d}-{val_loss:.4f}.hdf5'
early_stopping = EarlyStopping(monitor='val_loss',patience = 10,mode = 'auto')
# check_point = ModelCheckpoint(filepath=modelpath,monitor='val_loss',save_best_only=True,mode='auto')

# fit
hist = model.fit(x_train,y_train,batch_size = 64,validation_split = 0.2,epochs = 30,callbacks=[early_stopping,check_point])

# Evaluate
loss,acc = model.evaluate(x_test,y_test,batch_size = 64)
print('loss : ',loss)
print('accuracy: ',acc)

# visualization
import matplotlib.pyplot as plt
plt.figure(figsize = (10,6))
plt.subplot(211)    # 2 row 1 column
plt.plot(hist.history['loss'],marker = '.',c='red',label = 'loss')
plt.plot(hist.history['val_loss'],marker = '.',c='blue',label = 'val_loss')
plt.grid()

plt.title('Cost')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')

plt.subplot(212)    # 2 row 2 column
plt.plot(hist.history['acc'],marker = '.',c='red',label = 'acc')
plt.plot(hist.history['val_acc'],marker = '.',c='blue',label = 'val_acc')
plt.grid()

plt.title('Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(loc='upper right')
plt.show()
