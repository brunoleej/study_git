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

# x = x.reshape(6,4,1)
# y = y.reshape(6,1,1)

x_pred = dataset[-1:,1:]
# x_pred = x_pred.reshape(1,4,1)
# print(x_pred.shape)

print(x.shape,y.shape)

# model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense,Input,Activation

input1 = Input(shape=(4,))
dense1 = Dense(40)(input1)
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
model.fit(x,y,epochs = 100, batch_size = 1,callbacks=early_stopping)

loss,mae = model.evaluate(x,y)
print('loss: ',loss)
print('mae: ', mae)

y_pred = model.predict(x_pred)
print('y_pred: ',y_pred)

# LSTM
# loss:  0.25046786665916443
# mae:  0.496021032333374
# y_pred:  [[9.765697]]

# Dense
# loss:  0.01980692334473133
# mae:  0.12011400610208511
# y_pred:  [[11.2961035]]