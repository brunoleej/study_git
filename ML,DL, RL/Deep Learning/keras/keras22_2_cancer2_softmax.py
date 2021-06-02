# keras21_cancer1.py 를 다중분류로 코딩하시오.

import numpy as np
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()

x = cancer.data
y = cancer.target
print(x.shape,y.shape)  # (569, 30) (569,)

# train_test_split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,train_size = 0.7,shuffle = True)

# preprocessing OneHotEncoding
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# model
from tensorflow.keras.layers import Dense,Input,Activation
from tensorflow.keras.models import Model
input1 = Input(shape = (30, ))
dense1 = Dense(64)(input1)
dense1 = Activation('relu')(dense1)
dense1 = Dense(128)(dense1)
dense1 = Dense(2)(dense1)
dense1 = Activation('softmax')(dense1)
model = Model(inputs = input1, outputs = dense1)

# compile
model.compile(loss = 'categorical_crossentropy',optimizer = 'adam',metrics = ['accuracy'])

# EarlyStopping
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor = 'accuracy',patience=30,mode = 'auto')

# fitting
model.fit(x_train,y_train,epochs = 100,validation_split = 0.2,callbacks = early_stopping)

# evaluate
loss = model.evaluate(x_test,y_test)
print(loss)

# predict
y_pred = model.predict(x[-5:-1])
print(y_pred)
print(y[-5:-1])