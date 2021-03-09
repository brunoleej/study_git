import numpy as np
import tensorflow as tf
from keras.datasets import fashion_mnist

(train_images,train_labels),(test_images,test_labels) = fashion_mnist.load_data()
print(train_images.shape,test_images.shape) # (60000, 28, 28) (10000, 28, 28)
print(np.min(train_images),np.max(test_images)) # 0 255
train_images, test_images = train_images / 255.0, test_images / 255.0
train_images = train_images.reshape(60000,784)
test_images = test_images.reshape(10000,784)

# to_categorical
from tensorflow.keras.utils import to_categorical
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# modeling
from tensorflow.keras.layers import Conv2D,Dense,MaxPool2D,Flatten,Dropout,Activation,Input
from tensorflow.keras.models import Model

input1 = Input(shape =(784,))
dense = Dense(500)(input1)
dense = Dense(500)(dense)
dense = Activation('relu')(dense)
dense = Dense(250)(dense)
dense = Dense(250)(dense)
dense = Activation('relu')(dense)
dense = Dense(120)(dense)
dense = Dense(120)(dense)
dense = Activation('relu')(dense)
dense = Dense(60)(dense)
dense = Dense(60)(dense)
dense = Activation('relu')(dense)
dense = Dense(30)(dense)
dense = Dense(30)(dense)
dense = Activation('relu')(dense)
dense = Dense(10)(dense)
dense = Activation('softmax')(dense)

model = Model(inputs = input1,outputs = dense,name = 'Dense_net')

# compile
model.compile(loss = 'categorical_crossentropy',optimizer = 'adam',metrics = ['acc'])

# fit
model.fit(train_images,train_labels,epochs = 5, batch_size = 64)

# evaluate
loss,acc = model.evaluate(test_images,test_labels,batch_size = 64)
print('loss: ',loss)
print('accuracay: ',acc)

# CNN
# loss:  0.20688886940479279
# acc:  0.9332000112533569

# Dense
# loss:  0.39355993270874023
# accuracay:  0.862500011920929