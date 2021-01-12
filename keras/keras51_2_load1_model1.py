# keras40_mnist2_cnn.py cope()

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from keras.datasets import mnist

(x_train,y_train),(x_test,y_test) = mnist.load_data()
print(x_train.shape,y_train.shape)  # (60000, 28, 28) (60000,)
print(x_test.shape,y_test.shape)    # (10000, 28, 28) (10000,)    
print(np.min(x_train),np.max(x_test))   # 0 255

print(x_train[0])
print('y_train[0] : ',y_train[0])
print(x_train[0].shape) # (28, 28)

# plt.imshow(x_train[0],'jet')
# plt.colorbar()
# plt.show()

# Preprocessing
x_train = tf.cast(x_train,dtype='float32')
x_train = x_train[...,tf.newaxis]
x_train = x_train / 255.
print(x_train[0].shape) # (28, 28, 1)

x_test = x_test[...,tf.newaxis]
x_test = x_test/255.
# (x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2], 1))
test_image = x_test[0,:,:,0]    # (28, 28)
print(test_image.shape)

# OneHotEncoding(to_categorical)
# from keras.utils import to_categorical

# Model Declaration
from keras.layers import Conv2D,MaxPool2D,Dense,Flatten,Dropout,Input,Activation
from keras.models import Model,load_model

# # Fully Connected
# input1 = Input(shape=(x_train[0].shape))
# net = Conv2D(32,3,padding='SAME')(input1)
# net = Activation('relu')(net)
# net = Conv2D(32,3,3,padding = 'SAME')(net)
# net = Activation('relu')(net)
# net = MaxPool2D(pool_size=(2,2))(net)
# net = Dropout(0.25)(net)

# net = Conv2D(64,3,padding='SAME')(net)
# net = Activation('relu')(net)
# net = Conv2D(64,3,3,padding = 'SAME')(net)
# net = Activation('relu')(net)
# net = MaxPool2D(pool_size=(2,2))(net)
# net = Dropout(0.25)(net)

# net = Flatten()(net)
# net = Dense(512)(net)
# net = Activation('relu')(net)
# net = Dropout(0.5)(net)
# net = Dense(10)(net)
# net = Activation('softmax')(net)
# model = Model(inputs = input1, outputs = net, name = 'CNN_Model')
model = load_model('../model/k51_1_model1.h5')

model.summary()
# EarlyStopping
from keras.callbacks import EarlyStopping,ModelCheckpoint
modelpath = '../modelCheckpoint/k45_mnist_{epoch:02d}-{val_loss:.4f}.hdf5'
early_stopping = EarlyStopping(monitor='val_loss',patience = 5,mode = 'auto')
check_point = ModelCheckpoint(filepath=modelpath,monitor='val_loss',save_best_only=True,mode='auto')

# Compile
model.compile(loss = 'sparse_categorical_crossentropy',optimizer = 'adam',metrics = ['acc'])

# fit
hist = model.fit(x_train,y_train,epochs = 30,validation_split=0.2,batch_size = 64,callbacks=[early_stopping,check_point])

# Evaluate
loss = model.evaluate(x_test,y_test,batch_size = 64)
print('loss : ',loss[0])
print('accuracy: ',loss[1])

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
plt.plot(hist.history['acc'],marker = '.',c='red')
plt.plot(hist.history['val_acc'],marker = '.',c='blue')
plt.grid()

plt.title('Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['accuracy','val_accuracy'])
plt.show()
