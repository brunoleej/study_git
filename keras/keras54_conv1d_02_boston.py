# LSTM와 비교
import numpy as np
import tensorflow as tf
from keras.datasets import fashion_mnist

# preprocessing
(train_images,train_labels),(test_images,test_labels) = fashion_mnist.load_data()
print(train_images.shape,test_images.shape) # (60000, 28, 28) (10000, 28, 28)
print(train_labels.shape,test_labels.shape) # (60000,) (10000,)
print(np.min(train_images),np.max(test_images)) # 0 255
train_images,test_images = train_images / 255.0, test_images / 255.0
print(np.min(train_images),np.max(test_images)) # 0.0 1.0
train_images = train_images[...,tf.newaxis]
test_images = test_images[...,tf.newaxis]
print(train_images.shape,test_images.shape) # (60000, 28, 28, 1) (10000, 28, 28, 1)

# to_categorical
from tensorflow.keras.utils import to_categorical
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Modeling
from tensorflow.keras.layers import Conv1D,MaxPool1D,Dense,Dropout,Activation,Flatten,Input
from tensorflow.keras.models import Model

# Fully Connected
input1 = Input(shape=(28,28,1))
net = Conv1D(32,3,padding='SAME')(input1)
net = Activation('relu')(net)
net = Conv1D(32,3,padding='SAME')(net)
net = Activation('relu')(net)
net = MaxPool1D(pool_size = 1)(net)
net = Dropout(0.25)(net)

net = Conv1D(64,3,padding='SAME')(net)
net = Activation('relu')(net)
net = Conv1D(64,3,padding='SAME')(net)
net = Activation('relu')(net)
net = MaxPool1D(pool_size = 1)(net)
net = Dropout(0.25)(net)

net = Flatten()(net)
net = Dense(512)(net)
net = Activation('relu')(net)
net = Dropout(0.5)(net)
net = Dense(10)(net)
net = Activation('softmax')(net)

model = Model(inputs = input1,outputs = net,name = 'fashion_mnist_CNN')

# Compile
model.compile(loss = 'categorical_crossentropy',optimizer='adam',metrics = ['acc'])

# fit
model.fit(train_images,train_labels,epochs=5,batch_size=64)

# evaluate
loss,acc = model.evaluate(test_images,test_labels,batch_size = 64)
print('loss: ', loss)
print('acc: ', acc)

# CNN
# loss:  0.20688886940479279
# acc:  0.9332000112533569
