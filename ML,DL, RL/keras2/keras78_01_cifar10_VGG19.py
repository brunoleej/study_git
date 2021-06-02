# VGG19

from tensorflow.keras.applications import VGG19
from tensorflow.keras.datasets import cifar10
import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

#1. DATA
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], x_train.shape[3])/255.
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], x_test.shape[3])/255.

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)
# (50000, 32, 32, 3) (50000, 10)
# (10000, 32, 32, 3) (10000, 10)

#2. Modeling

vgg19 = VGG19(weights='imagenet', include_top=False, input_shape=(32,32,3))

vgg19.trainable = False

model = Sequential()
model.add(vgg19)
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))

# model.summary()

#3. Compile, Train

lr = ReduceLROnPlateau(monitor='val_loss', factor=0.4, patience=10, verbose=1, mode='min')
es = EarlyStopping(monitor='val_loss', patience=20, mode='min')

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=1, callbacks=[lr, es])

loss, acc = model.evaluate(x_test, y_test, batch_size=32)
print("loss : ", loss)
print("acc : ", acc)

## CNN
# loss :  1.7849451303482056
# acc :  0.5971999764442444

###### 전이학습 ######
# VGG16 
# loss :  1.9435967206954956
# acc :  0.604200005531311

# VGG19
# loss :  1.9104373455047607
# acc :  0.5917999744415283