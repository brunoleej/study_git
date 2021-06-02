import numpy as np

a = np.array(range(1,11))
size = 5

def split_x(seq,size):   # 2 parameter
    aaa = []             # empty list
    for i in range(len(seq) - size + 1):    # 6 times loop
        subset = seq[i : (i + size)]  
        # aaa.append([item for item in subset])
        aaa.append(subset)
    print(type(aaa))
    return np.array(aaa)

dataset = split_x(a, size)

print('========================')
# print(dataset)

x = dataset[:,:4]
y = dataset[:,4:]
print(x.shape,y.shape)  # (6, 4) (6, 1)

x = x.reshape(6,4,1)

x_pred = dataset[-1:,1:]
print(x_pred,x_pred.shape)  # [[ 7  8  9 10]] (1, 4)

x_pred = x_pred.reshape(1,4,1)
print(x_pred.shape)

print(x.shape,y.shape)

# train_test_split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state=1)

# model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense,LSTM,Input,Activation

input1 = Input(shape=(4,1))
dense1 = LSTM(40)(input1)
dense1 = Activation('relu')(dense1)
dense1 = Dense(24)(dense1)
dense1 = Activation('relu')(dense1)
dense1 = Dense(1)(dense1)

model = Model(inputs = input1, outputs = dense1)

# EarlyStopping
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='mae',patience=10,mode='auto')

# compile
model.compile(loss = 'mse',optimizer = 'adam',metrics = ['mae'])
# fit
model.fit(x_train,y_train,epochs = 100, batch_size = 1,callbacks=early_stopping)

loss,mae = model.evaluate(x_test,y_test)
print('loss: ',loss)
print('mae: ', mae)

y_pred = model.predict(x_pred)
print('y_pred: ',y_pred)

# LSTM
# loss:  0.25046786665916443
# mae:  0.496021032333374
# y_pred:  [[9.765697]]
