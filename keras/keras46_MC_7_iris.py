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

# Earlystooping, ModelCheckpoint
from keras.callbacks import EarlyStopping,ModelCheckpoint
# modelpath = './modelCheckpoint/k45_iris_data_{epoch:02d}-{val_loss:.4f}.hdf5'
early_stopping = EarlyStopping(monitor='val_loss',patience = 20,mode = 'auto')
# check_point = ModelCheckpoint(filepath=modelpath,monitor='val_loss',save_best_only=True,mode='auto')

hist = model.fit(x_train,y_train,epochs = 3000, validation_split =0.2,callbacks = [early_stopping,check_point])

# Evaluate
loss,acc = model.evaluate(x_test,y_test)
print('loss: ',loss)
print('accuracy: ',acc)

# prediction
y_pred = model.predict(x[-5:-1])
print(y_pred)
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