import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.utils import shuffle
from tensorflow.python.ops.gen_math_ops import Min

# data
cancer = load_breast_cancer() 

print(cancer.DESCR)
print(cancer.feature_names)

x = cancer.data
y = cancer.target
print(x.shape)  # (569,30)
print(y.shape)  # (569, )
print(x[:5])
print(y)

# train_test_split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,train_size = 0.7,random_state = 1)

# 전처리 / minmax, train_test_split
# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# scaler.fit(x_train)
# x_train = np.transpose(x_train)
# x_test = np.transpose(x_test)
# x_val = np.transpose(x_val)

# modeling
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense,Activation,Input

input1 = Input(shape = (30,))
dense1 = Dense(300)(input1)
dense1 = Activation('relu')(dense1)
dense1 = Dense(150)(dense1)
dense1 = Activation('relu')(dense1)
dense1 = Dense(60)(dense1)
dense1 = Activation('sigmoid')(dense1)
dense1 = Dense(1)(dense1)
model = Model(inputs= input1, outputs = dense1)

# compile, fit
    # mean_squared_error
model.compile(loss = 'binary_crossentropy',optimizer='adam',metrics=['acc'])

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor = 'loss',patience = 20,mode = 'auto')

hist = model.fit(x_train,y_train,epochs = 1000,validation_split = 0.2,callbacks=[early_stopping])

# loss = model.evaluate(x_test,y_test)
# print(loss)

# 실습 1. acc.0.985 이상 올릴것
# 실습 2. predict 출력해볼것
# y_pred = model.predict(x[-5:-1])
# print(y[-5:-1])

print(hist)
print(hist.history.keys())  
# compile not in metrics dict_keys(['loss', 'val_loss'])
# compile in metrics dict_keys(['loss', 'acc', 'val_loss', 'val_acc'])

print(hist.history['loss'])

import matplotlib.pyplot as plt
plt.plot(hist.history['loss'],label = 'loss')
plt.plot(hist.history['acc'],label='acc')
plt.plot(hist.history['val_loss'],label='val_loss')
plt.plot(hist.history['val_acc'],label = 'val_acc')
plt.title('loss & acc')
plt.ylabel('loss, acc')
plt.xlabel('epochs')
plt.legend()
plt.show()