# Multi-Class Classification
# Softmax
# Categorical_crossentropy

from sklearn.datasets import load_wine

wine = load_wine()
# print(wine.DESCR)
# print(wine.feature_names)

x = wine.data
y = wine.target
print(x,y)
print(x.shape,y.shape)  # (178, 13) (178,)

# train_test_split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3)

print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)    # (124, 13) (54, 13) (124,) (54,)

# OneHotEncoding
from keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train.shape,y_test.shape)   # (124, 3) (54, 3)

# # preprocessing
# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# scaler.fit()

# DNN model
from tensorflow.keras.layers import Dense,Input,Activation
from tensorflow.keras.models import Model
input1 = Input(shape = (13, ))
dense1 = Dense(64)(input1)
dense1 = Activation('relu')(dense1)
dense1 = Dense(128)(dense1)
dense1 = Dense(3)(dense1)
dense1 = Activation('softmax')(dense1)

model = Model(inputs = input1, outputs = dense1)

# compile
model.compile(loss = 'categorical_crossentropy',optimizer = 'adam',metrics = ['accuracy'])

# EarlyStopping
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor = 'accuracy',patience=30,mode = 'auto')

# fitting
hist = model.fit(x_train,y_train,epochs = 1000,validation_split = 0.2,callbacks = [early_stopping])

# # evaluate
# loss,accuracy = model.evaluate(x_test,y_test)
# print('loss: ',loss)
# print('accuracy: ',accuracy)

# # predict
# y_pred = model.predict(x[-5:-1])
# # print(y_pred)
# print(y[-5:-1])

# Dense
# loss:  1.1334611177444458
# accuracy:  0.7777777910232544

print(hist)
print(hist.history.keys())  
# compile not in metrics dict_keys(['loss', 'val_loss'])
# compile in metrics dict_keys(['loss', 'acc', 'val_loss', 'val_acc'])

print(hist.history['loss'])

import matplotlib.pyplot as plt
plt.plot(hist.history['loss'],label = 'loss')
plt.plot(hist.history['accuracy'],label='accuracy')
plt.plot(hist.history['val_loss'],label='val_loss')
plt.plot(hist.history['val_accuracy'],label = 'val_accuracy')
plt.title('loss & acc')
plt.ylabel('loss, acc')
plt.xlabel('epochs')
plt.legend()
plt.show()