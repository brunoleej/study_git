import numpy as np

a = np.array(range(1,11))

from tensorflow.keras.models import load_model
model = load_model('./model/save_keras35.h5')
## test ##########################
from tensorflow.keras.layers import Dense
model.add(Dense(5,name = 'su_dense1'))   # layer name : dense
model.add(Dense(1,name = 'su_dense2'))   # laer name : dense_1
##################

model.summary()

print(a.shape)  # (10,)
print(a)
size = 6

# LSTM 모델을 구성하시오

def split_x(seq, size) :
    aaa = []  
    for i in range(len(seq) - size + 1) :       # range(len(seq) - size + 1) : 반복횟수(= 행의 개수), # size : 열의 개수
        subset = seq[i : (i+size)]
        aaa.append(subset)
    # print(type(aaa))
    return np.array(aaa)

dataset = split_x(a, size)  # (6, 5)
print(dataset.shape)    # a size 4 : (7, 4)
print(dataset)

x = dataset[:,:5]
y = dataset[:,5:]
print(x,'\n',y)
print(x.shape,y.shape)  # (5, 5) (5, 1)

# input reshape
x = x.reshape(5,5,1)

# predict value
x_pred = dataset[-1:,1:]    
print(x_pred.shape) # (1, 5)

# predict value reshape
x_pred = x_pred.reshape(1,5,1)
print(x_pred.shape)
print(x.shape,y.shape)

# train_test_split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state=1)

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

# loss:  0.19780848920345306
# mae:  0.34719300270080566
# y_pred:  [[8.60549]]
