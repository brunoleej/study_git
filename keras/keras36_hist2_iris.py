import numpy as np
from sklearn.datasets import load_iris

# 1. Data
# x,y = load_iris(return_X_y=True)
iris = load_iris()
x = iris.data
y = iris.target

print(iris.DESCR)
print(iris.feature_names)
print(x.shape,y.shape)  # (150, 4) (150,)
print(x[:5])
print(y)

# preprocessing
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


from tensorflow.keras.layers import Dense,Input,Activation
from tensorflow.keras.models import Model

input1 = Input(shape=(4,))
dense1 = Dense(64)(input1)
dense1 = Activation('relu')(dense1)
dense1 = Dense(128)(dense1)
dense1 = Dense(3)(dense1)
dense1 = Activation('softmax')(dense1)

model = Model(inputs = input1,outputs = dense1)

model.compile(loss = 'categorical_crossentropy',optimizer = 'adam',metrics = ['acc'])

# EarlyStopping
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor = 'accuracy',patience=20,mode = 'auto')

hist = model.fit(x_train,y_train,epochs = 1000, validation_split =0.2,callbacks = [early_stopping])

# loss = model.evaluate(x_test,y_test)
# print(loss)

# y_pred = model.predict(x[-5:-1])
# print(y_pred)
# print(y[-5:-1])

print(hist)
print(hist.history.keys())  
# compile not in metrics dict_keys(['loss', 'val_loss'])
# compile in metrics dict_keys(['loss', 'acc', 'val_loss', 'val_acc'])

print(hist.history['loss'])

import matplotlib.pyplot as plt
plt.plot(hist.history['loss'],label = 'epochs')
plt.plot(hist.history['val_loss'],label='val_loss')
plt.plot(hist.history['acc'],label='acc')
plt.plot(hist.history['val_acc'],label = 'val_acc')
plt.title('loss & acc')
plt.ylabel('loss, acc')
plt.xlabel('epochs')
plt.legend()
plt.show()