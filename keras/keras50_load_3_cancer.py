import numpy as np

data = np.load('./data/cancer_data.npy')
target = np.load('./data/cancer_target.npy')

# train_test_split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(data,target,train_size = 0.7,random_state = 1)

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

# EarlyStopping
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor = 'acc',patience = 20,mode = 'auto')

# fit
model.fit(x_train,y_train,epochs = 300,validation_split = 0.2,callbacks=[early_stopping])

# Evaluate
loss,acc = model.evaluate(x_test,y_test)
print('loss: ',loss)
print('accuracy: ',acc)

# loss:  0.21281705796718597
# accuracy:  0.9064327478408813
