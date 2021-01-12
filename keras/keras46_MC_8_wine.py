import numpy as np
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
dense1 = Dense(128)(input1)
dense1 = Dense(128)(dense1)
dense1 = Activation('relu')(dense1)
dense1 = Dense(64)(dense1)
dense1 = Dense(64)(dense1)
dense1 = Activation('relu')(dense1)
dense1 = Dense(3)(dense1)
dense1 = Activation('softmax')(dense1)
model = Model(inputs = input1, outputs = dense1)

# compile
model.compile(loss = 'categorical_crossentropy',optimizer = 'adam',metrics = ['accuracy'])

# Earlystooping, ModelCheckpoint
from keras.callbacks import EarlyStopping,ModelCheckpoint
modelpath = '../modelCheckpoint/k45_iris_data_{epoch:02d}-{val_loss:.4f}.hdf5'
early_stopping = EarlyStopping(monitor='val_loss',patience = 20,mode = 'auto')
check_point = ModelCheckpoint(filepath=modelpath,monitor='val_loss',save_best_only=True,mode='auto')

# fit
hist = model.fit(x_train,y_train,epochs = 3000,validation_split = 0.2,callbacks = [early_stopping,check_point])

# Evaluate
loss,accuracy = model.evaluate(x_test,y_test)
print('loss: ',loss)
print('accuracy: ',accuracy)

# Prediction
y_pred = model.predict(x[-5:-1])
# print(y_pred)
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