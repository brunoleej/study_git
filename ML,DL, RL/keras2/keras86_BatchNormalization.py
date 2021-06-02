# kernel_initializer='he_normal'
# BatchNormalization()
# kernel_regularizer

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist

#1. DATA
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# print(x_train.shape, y_train.shape) # (60000, 28, 28)--> 흑백 1 생략 가능 (60000,) 
# print(x_test.shape, y_test.shape)   # (10000, 28, 28)                     (10000,)  > 0 ~ 9 다중 분류

# plt.imshow(x_train[0], 'gray')  # 0 : black, ~255 : white (가로 세로 색깔)
# plt.imshow(x_train[0]) # 색깔 지정 안해도 나오긴 함
# plt.show()  

# preprocessing
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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.layers import BatchNormalization, Activation
from tensorflow.keras.regularizers import l1,l2,l1_l2

model = Sequential()
model.add(Conv2D(filters=16, kernel_size=(2,2), padding='same', strides=1, input_shape=(28,28,1)))
model.add(BatchNormalization())
model.add(Activation('relu'))

# model.add(MaxPooling2D(pool_size=2))
# model.add(Dropout(0.1))
model.add(Conv2D(32,(2,2), kernel_initializer='he_normal')) 
'''
[kernel_initializer = weight 초기화]
    he_normal             : relu, elu, selu, 
    Xavier Initialization : sigmoid, tanh

[bias_initializer]
    예전에는 bias를 0.01 혹은 0.1 정도로 사용했지만, 최근에는 0으로 초기화한다.
    bias_initializer='zeros'

[kernel_regularizer] 
    과적합을 예방하고 일반화하는 성능을 높인다.
    l1 : 가중치의 절댓값에 비례하는 비용이 추가됨
    l2 : 가중치의 제곱에 비례하는 비용이 추가됨

[BathNormalization]
    평균과 분산을 조정하는 과정이 신경망 안에 포함되어 학습 시 평균과 분산을 조정하는 과정이 같이 조절된다.

[Dropout]
    입력 노드 중 일부를 무작위로 0으로 바꿈
'''
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv2D(32, (2,2), kernel_regularizer=l1(l1=0.01)))
'''
kernel_regularizer : 정규화하는 방법
'''
model.add(Dropout(0.2))

model.add(Conv2D(32, (2,2), strides=2))
model.add(MaxPooling2D(pool_size=2))

model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))

# model.summary()

# Compile, Train

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# 체크포인트의 가중치를 저장할 파일경로 지정
# modelpath='./modelCheckpoint/k45_mnist_{epoch:02d}-{val_loss:.4f}.hdf5'
es = EarlyStopping(monitor='val_loss', patience=5, mode='max')
# cp = ModelCheckpoint(filepath=modelpath,monitor='val_loss', save_best_only=True, mode='auto')

                    
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
hist = model.fit(x_train, y_train, epochs=15, batch_size=32, validation_split=0.2, callbacks=[es])#, cp])

# Evaluate, Predict
result = model.evaluate(x_test, y_test, batch_size=32)
print("loss : ", result[0])
print("accuracy : ", result[1])


# 응용
# y_test 10개와 y_test 10개를 출력하시오

# print("y_test[:10] :\n", y_test[:10])
print("y_test[:10] :")
print(np.argmax(y_test[:10],axis=1))

y_predict = model.predict(x_test[:10])
print("y_pred[:10] :")  
print(np.argmax(y_predict,axis=1))

# 시각화
# import matplotlib.pyplot as plt

# plt.figure(figsize=(10,6))  # 판 사이즈 (가로 10, 세로 6)

# plt.subplot(2, 1, 1)    # plot : 도화지 하나에 그림을 그리겠다.
#                         # 2행 1열 중 첫 번째
#                         # 만약 (3, 1, 1) 이라면 세 개의 plot이 있어야 한다. (3, 1, 1) (3, 1, 2) (3, 1, 3)
# plt.plot(hist.history['loss'], marker='.', c='red', label='loss')   # label=' ' >> legend에서 설정한 위치에 라벨이 표시된다.
# plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
# plt.grid()

# # plt.title('손실비용') # 과제 : 한글 깨짐 오류 해결할 것
# plt.title('Cost Loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(loc='upper right')   # loc 를 명시하지 않으면 그래프가 비어있는 지역에 자동으로 위치한다.

# plt.subplot(2, 1, 2)    # 2행 1열 중 두 번째
# # plt.plot(hist.history['accuracy'], marker='.', c='red', label='accuracy')   # label=' ' >> legend에서 설정한 위치에 라벨이 표시된다.
# # plt.plot(hist.history['val_accuracy'], marker='.', c='blue', label='val_accuracy')
# plt.plot(hist.history['accuracy'], marker='.', c='red')   
# plt.plot(hist.history['val_accuracy'], marker='.', c='blue')
# plt.grid()              # 모눈종이 격자위에 그리겠다.

# # plt.title('정확도')   # 과제 : 한글 깨짐 오류 해결할 것
# plt.title('Accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# # plt.legend(loc='upper right')
# plt.legend(['accuracy','val_accuracy'])

# plt.show()

# loss :  0.034563612192869186
# acc :  0.9889000058174133
# y_test[:10] :
# [7 2 1 0 4 1 4 9 5 9]
# y_pred[:10] :
# [7 2 1 0 4 1 4 9 5 9]