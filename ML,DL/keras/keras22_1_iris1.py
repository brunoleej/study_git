import numpy as np
from sklearn.datasets import load_iris

# Data
iris = load_iris()
x = iris.data
y = iris.target

print(iris.DESCR)
print(iris.feature_names)
print(x.shape,y.shape)
print(x[:5])
print(y)

# Preprocessing
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,train_size = 0.7,shuffle = True)

# OneHotEncoding
# from sklearn.preprocessing import OneHotEncoder

from tensorflow.keras.utils import to_categorical
# from keras.utils.np_utils import to_categorical

# y = to_categorical(y)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(y_train)
print(x_train.shape)  # (105,4)
print(y_train.shape)  # (105,3)

# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# scaler.fit(x_train)
# x_train = np.transpose(x_train)
# print(x_train.shape)
# print(y_train.shape)
# x_test = np.transpose(x_test)
# x_val = np.transpose(x_val)


# Model
from tensorflow.keras.layers import Dense,Input,Activation
from tensorflow.keras.models import Model

input1 = Input(shape=(4,))
dense1 = Dense(128)(input1)
dense1 = Dense(128)(dense1)
dense1 = Activation('relu')(dense1)
dense1 = Dense(64)(dense1)
dense1 = Dense(64)(dense1)
dense1 = Dense(3)(dense1)
dense1 = Activation('softmax')(dense1)
model = Model(inputs = input1,outputs = dense1)

# Model Compile
model.compile(loss = 'categorical_crossentropy',optimizer = 'adam',metrics = ['acc'])

# EarlyStopping
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor = 'acc',patience=20,mode = 'auto')

model.fit(x_train,y_train,epochs = 100, validation_split =0.2,callbacks = [early_stopping])

# Evaluate
loss,acc = model.evaluate(x_test,y_test)
print('loss: ',loss)
print('accuracy: ',acc)

# prediction
y_pred = model.predict(x[-5:-1])
print(y_pred)
print(y[-5:-1])

# 1
# loss:  0.26552248001098633
# accuracy:  0.9111111164093018