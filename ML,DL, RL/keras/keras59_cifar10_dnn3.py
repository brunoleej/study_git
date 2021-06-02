# 다차원 Dense 모델
# (n,32,32,3) -> (n,32,32,3)
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np

# DATA
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(x_train.shape, y_train.shape) # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)   # (10000, 32, 32, 3) (10000, 1)

# print(x_train[0])   
# print("y_train[0] : " , y_train[0])   # 6
# print(x_train[0].shape)               # (32, 32, 3)

# print(np.min(x_train),np.max(x_train))  # 0 ~ 255

# x > preprocessing
# x_train = x_train.reshape(x_train.shape[0],x_train.shape[1]*x_train.shape[2]*x_train.shape[3]) / 255.
# x_test = x_test.reshape(x_test.shape[0],x_test.shape[1]*x_test.shape[2]*x_test.shape[3]) / 255.

# y > preprocessing
# from tensorflow.keras.utils import to_categorical
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)

# print(y_train.shape)    # (50000, 10)
# print(y_test.shape)     # (10000, 10)

y_train = x_train 
y_test = x_test  
print(y_train.shape)    # (50000, 32, 32, 3)
print(y_test.shape)     # (10000, 32, 32, 3)

#2. Modeling
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten

model = Sequential()
model.add(Dense(256,input_shape=(32,32,3), activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(3,activation='softmax'))

model.summary()
#  (None, 32, 32, 256)  -->  (None, 32, 32, 3) 

'''
#3. Compile, Train
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss',patience=10,mode='min')
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
model.fit(x_train, y_train, epochs=10, batch_size=32,validation_split=0.2, verbose=1,callbacks=[es])
#4. predict, Evaluate
loss, acc = model.evaluate(x_test, y_test, batch_size=32)
print("loss : ", loss)
print("acc : ", acc)
print("y_test : ", np.argmax(y_test[-5:-1],axis=1))
y_pred = model.predict(x_test[-5:-1])
print("y_pred : ", np.argmax(y_pred,axis=1))
# DNN
# loss :  1.6483410596847534
# acc :  0.41110000014305115
# y_test :  [8 3 5 1]
# y_pred :  [8 6 5 6]
'''
