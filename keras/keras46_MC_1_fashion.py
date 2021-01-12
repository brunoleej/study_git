# CNN
# fashion_mnist
# from tensorflow.keras.callbacks import ModelCheckpoint

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist

#1. DATA
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
print(x_train.shape, y_train.shape) # (60000, 28, 28)--> 흑백 1 생략 가능 (60000,) 
print(x_test.shape, y_test.shape)   # (10000, 28, 28)                     (10000,)

# print(x_train[0])   
# print("y_train[0] : " , y_train[0])   # 9
# print(x_train[0].shape)               # (28, 28)

# plt.imshow(x_train[0], 'gray')        # 0 : black, ~255 : white (가로 세로 색깔)
# # plt.imshow(x_train[0]) # 색깔 지정 안해도 나오긴 함
# plt.show()  

# x > preprocessing
# print(np.min(x_train),np.max(x_train))  # 0 ~ 255
x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],1)/255.
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],1)/255.

print(x_train.shape)    # (60000, 28, 28, 1)
print(x_test.shape)     # (10000, 28, 28, 1)
print(np.min(x_train),np.max(x_train))  # 0.0 ~ 1.0

# y > preprocessing
# print(y_train[:20]) # 0 ~ 9
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(y_train.shape)    # (60000, 10)
print(y_test.shape)     # (10000, 10)

#2. Modeling
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout

model = Sequential()
model.add(Conv2D(filters=112, kernel_size=(2,2),padding='same',strides=1,input_shape=(x_train.shape[1],x_train.shape[2],x_train.shape[3])))
model.add(MaxPool2D(pool_size=2))
model.add(Dropout(0.2))
model.add(Conv2D(filters=84,kernel_size=(2,2)))
model.add(MaxPool2D(pool_size=2))
model.add(Dropout(0.2))
model.add(Conv2D(filters=28,kernel_size=(2,2)))
model.add(Conv2D(filters=28,kernel_size=(2,2)))
model.add(Flatten())
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10,activation='softmax'))

# model.summary()

#3. Compile, Train
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# 체크포인트의 가중치를 저장할 파일경로 지정
modelpath='../modelcheckpoint/k46_1_fashion_{epoch:02d}-{val_loss:.4f}.hdf5'
                                        # 02d : 정수 두 자리만 적겠다. / .4f : 소수점 아래 4째자리까지 적겠다.
                                        # 저장 예시) k45_mnist_37-0.0100.hdf5
                                        # 저장된 파일 중에 가장 마지막에 생성된게 가장 좋은 것이 됨
es = EarlyStopping(monitor='val_loss', patience=5, mode='max')
cp = ModelCheckpoint(filepath=modelpath,monitor='val_loss', save_best_only=True, mode='auto')
                    # filepath : 최저점이 찍힐 때마다 가중치가 세이브된 파일이 생성된다. 
                    # 궁극의 목적 : 최적의 weight를 구하기 위해서
                    # predict할 때 혹은 evaluate 할 때 이 weight를 넣기만 하면된다.
                    
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
hist = model.fit(x_train, y_train, epochs=15, batch_size=32, validation_split=0.2, callbacks=[es, cp])

#4. Evaluate, predict
loss, acc = model.evaluate(x_test, y_test, batch_size=32)
print("loss : ", loss)
print("acc : ", acc)

print("y_test : ", np.argmax(y_test[-5:-1],axis=1))
y_pred = model.predict(x_test[-5:-1])
print("y_pred : ", np.argmax(y_pred,axis=1))
