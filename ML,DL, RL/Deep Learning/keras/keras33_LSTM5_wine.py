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

# reshape
x = x.reshape(178,13,1)

# train_test_split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,shuffle=True)

print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)    # (124, 13) (54, 13) (124,) (54,)
print(x_train.shape,y_train.shape)  # (124, 13, 1) (124,)

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
from tensorflow.keras.layers import Dense,Input,Activation,LSTM
from tensorflow.keras.models import Model
input1 = Input(shape = (13,1))
dense1 = LSTM(64)(input1)
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
model.fit(x_train,y_train,epochs = 1000,validation_split = 0.2,callbacks = [early_stopping])

# evaluate
loss,accuracy = model.evaluate(x_test,y_test)
print('loss: ',loss)
print('accuracy: ',accuracy)

# predict
y_pred = model.predict(x[-5:-1])
# print(y_pred)
print(y[-5:-1])

# Dense
# loss:  1.1334611177444458
# accuracy:  0.7777777910232544

# LSTM
# loss:  0.21899166703224182
# accuracy:  0.9444444179534912