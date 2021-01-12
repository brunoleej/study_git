import numpy as np

data = np.load('./data/wine_data.npy')
target = np.load('./data/wine_target.npy')

# train_test_split
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(data,target,test_size = 0.3)
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

# EarlyStopping
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor = 'accuracy',patience=30,mode = 'auto')

# fit
model.fit(x_train,y_train,epochs = 1000,validation_split = 0.2,callbacks = [early_stopping])

# Evaluate
loss,accuracy = model.evaluate(x_test,y_test)
print('loss: ',loss)
print('accuracy: ',accuracy)

# loss:  2.435310125350952
# accuracy:  0.5370370149612427

# Prediction
y_pred = model.predict(data[-5:-1])
# print(y_pred)
print(target[-5:-1])


