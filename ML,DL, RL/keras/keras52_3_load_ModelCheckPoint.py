# ModelCheckPoint >> weight가 저장되어 있다.
# 체크포인트에 저장되어 있는 weight를 불러오는 게 더 좋은 결과가 나온다.

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist

#1. DATA
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,) 
# print(x_test.shape, y_test.shape)   # (10000, 28, 28) (10000,)  > 0 ~ 9 다중 분류

# x >> preprocessing
x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255. 
x_test = x_test.reshape(10000, 28, 28, 1)/255. 

# y >> OnHotEncoding
from sklearn.preprocessing import OneHotEncoder

y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
# print(y_train[0])       # [5]
# print(y_train.shape)    # (60000, 1)
# print(y_test[0])        # [7]
# print(y_test.shape)     # (10000, 1)

encoder = OneHotEncoder()
encoder.fit(y_train)
encoder.fit(y_test)
y_train = encoder.transform(y_train).toarray()  #toarray() : list 를 array로 바꿔준다.
y_test = encoder.transform(y_test).toarray()    #toarray() : list 를 array로 바꿔준다.
# print(y_train)
# print(y_test)
# print(y_train.shape)    # (60000, 10)
# print(y_test.shape)     # (10000, 10)

#2. Modling
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

# model = Sequential()
# model.add(Conv2D(filters=16, kernel_size=(2,2), padding='same', strides=1, input_shape=(28,28,1)))
# model.add(MaxPooling2D(pool_size=2))
# model.add(Dropout(0.1))
# model.add(Conv2D(filters=16, kernel_size=(4,4), padding='same', strides=1))
# model.add(MaxPooling2D(pool_size=3))
# model.add(Dropout(0.1))
# model.add(Flatten())
# model.add(Dense(8))
# model.add(Dense(10, activation='softmax'))

# (1) 모델링 하고 난 직후 model.save
# model.save('../data/h5/k52_1_model1.h5')

# k52_1_MCK_0.0589.hdf5 와 비교

#3 Compile, Train
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# modelpath='../data/modelcheckpoint/k52_1_mnist_{epoch:02d}-{val_loss:.4f}.hdf5'
# es = EarlyStopping(monitor='val_loss', patience=5, mode='max')
# cp = ModelCheckpoint(filepath=modelpath,monitor='val_loss', save_best_only=True, mode='auto')

# model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
# hist = model.fit(x_train, y_train, epochs=15, batch_size=32, validation_split=0.2, callbacks=[es]) #, cp])

# (2) 컴파일, 훈련한 후 model.save
# model.save('../data/h5/k52_1_model2.h5')
# model.save_weights('../data/h5/k52_1_weight.h5')

# Evaluate, Predict

# ====== load_weight vs load_model 비교 ======

# model 1 : 
# fit한 부분만 뺀 상황 - 모델과 컴파일은 필요함
# 저장한 가중치(훈련의 결과)를 불러오므로 fit을 할 필요가 없다.
# model.load_weights('../data/h5/k52_1_weight.h5')

# result = model.evaluate(x_test, y_test, batch_size=32)
# print("가중치_loss : ", result[0])
# print("가중치_accuracy : ", result[1])
# 결과가 바로 나온다.
# 가중치_loss :  0.057048145681619644
# 가중치_accuracy :  0.9811000227928162


# model 2 : 
# 모델과 가중치를 불러온다.
# 따로 모델, 컴파일, 훈련을 할 필요 없음
# model2 = load_model('../data/h5/k52_1_model2.h5')

# result2 = model2.evaluate(x_test, y_test, batch_size=32)
# print("로드모델_loss : ", result2[0])
# print("로드모델_accuracy : ", result2[1])
# 로드모델_loss :  0.057048145681619644
# 로드모델_accuracy :  0.9811000227928162

# model 3 : 
# checkpoint 를 불러온다.
# 이미 가중치가 저장되어 있기 때문에 훈련할 때마다 값이 변하지 않는다.
# 체크포인트로 결과나온 값이 더 좋다.
model = load_model('../data/modelcheckpoint/k52_1_mnist_05-0.0626.hdf5')

result = model.evaluate(x_test, y_test, batch_size=32)
print("로드체크포인트_loss : ", result[0])
print("로드체크포인트_accuracy : ", result[1])