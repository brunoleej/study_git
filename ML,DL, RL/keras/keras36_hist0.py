import numpy as np
from tensorflow.keras.layers import Dense,LSTM
from tensorflow.keras.models import  Sequential

#. Data
a = np.array(range(1,101))
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
print(dataset.shape)    # (96, 5)

x = dataset[:,:4]
y = dataset[:,-1]
print(x.shape,y.shape)  # (96, 4) (96,)

x = x.reshape(x.shape[0],x.shape[1],1)  
print(x.shape)  # (96, 4, 1)

# model
from tensorflow.keras.models import load_model
model = load_model('./model/save_keras35.h5')
model.add(Dense(5,name = 'jisu1'))
model.add(Dense(1, name = 'jisu2'))

# EarlyStopping
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor = 'loss',patience=10,mode='auto')

# compile, fit
model.compile(loss = 'mse',optimizer = 'adam',metrics = ['acc'])
hist = model.fit(x,y,epochs = 1000,batch_size = 32,verbose=1,validation_split = 0.2, callbacks=[early_stopping])

print(hist)
print(hist.history.keys())  
# compile not in metrics dict_keys(['loss', 'val_loss'])
# compile in metrics dict_keys(['loss', 'acc', 'val_loss', 'val_acc'])

print(hist.history['loss'])

import matplotlib.pyplot as plt
plt.plot(hist.history['loss'],label = 'epochs')
plt.plot(hist.history['val_loss'],label='val_loss')
plt.plot(hist.history['acc'],label='acc')
plt.plot(hist.history['val_acc'],label = 'val_acc')
plt.title('loss & acc')
plt.ylabel('loss, acc')
plt.xlabel('epochs')
plt.legend()
plt.show()