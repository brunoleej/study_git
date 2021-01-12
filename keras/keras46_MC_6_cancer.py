# Classification Problem
import numpy as np
from sklearn.datasets import load_breast_cancer

# Data
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
dense1 = Dense(300)(dense1)
dense1 = Activation('relu')(dense1)
dense1 = Dense(150)(dense1)
dense1 = Dense(150)(dense1)
dense1 = Activation('relu')(dense1)
dense1 = Dense(60)(dense1)
dense1 = Dense(60)(dense1)
dense1 = Activation('relu')(dense1)
dense1 = Dense(30)(dense1)
dense1 = Dense(30)(dense1)
dense1 = Activation('relu')(dense1)
dense1 = Dense(1)(dense1)
dense1 = Activation('sigmoid')(dense1)
model = Model(inputs= input1, outputs = dense1)

# compile
model.compile(loss = 'binary_crossentropy',optimizer='adam',metrics=['acc'])

# Earlystooping, ModelCheckpoint
from keras.callbacks import EarlyStopping,ModelCheckpoint
# modelpath = './modelCheckpoint/k45_breast_cancer_{epoch:02d}-{val_loss:.4f}.hdf5'
early_stopping = EarlyStopping(monitor='val_loss',patience = 20,mode = 'auto')
# check_point = ModelCheckpoint(filepath=modelpath,monitor='val_loss',save_best_only=True,mode='auto')

# fit
hist = model.fit(x_train,y_train,epochs = 300,validation_split = 0.2,callbacks=[early_stopping,check_point])

# Evaluate
loss,acc = model.evaluate(x_test,y_test)
print('loss: ',loss)
print('accuracy: ',acc)

# 실습 1. acc.0.985 이상 올릴것
# 실습 2. predict 출력해볼것
y_pred = model.predict(x[-5:-1])
print(y[-5:-1])

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
