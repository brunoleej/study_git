# tensorflow.keras.layers >> Reshape 레이어
# model.add(Reshape((28,28,1)))

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist

#1. DATA
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# print(x_train.shape, y_train.shape) # (60000, 28, 28)--> 흑백 1 생략 가능 (60000,) 
# print(x_test.shape, y_test.shape)   # (10000, 28, 28)                     (10000,)  > 0 ~ 9 다중 분류

# print(x_train[0])   
# print("y_train[0] : " , y_train[0])   # 5
# print(x_train[0].shape)               # (28, 28)

# preprocessing
x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255. 
# 4차원 만들어준다. float타입으로 바꾸겠다. -> /255. -> 0 ~ 1 사이로 수렴됨
x_test = x_test.reshape(10000, 28, 28, 1)/255. 
# x_test.reshape(x_test.shape[0], x_test.shape[1],x_test.shape[2],1)

y_train = x_train 
y_test = x_test   

# print(y_train.shape)    # (60000, 28, 28, 1)
# print(y_test.shape)     # (10000, 28, 28, 1)      

'''
y >> OnHotEncoding
from sklearn.preprocessing import OneHotEncoder
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
print(y_train[0])       # [5]
print(y_train.shape)    # (60000, 1)
print(y_test[0])        # [7]
print(y_test.shape)     # (10000, 1)
encoder = OneHotEncoder()
encoder.fit(y_train)
encoder.fit(y_test)
y_train = encoder.transform(y_train).toarray()  #toarray() : list 를 array로 바꿔준다.
y_test = encoder.transform(y_test).toarray()    #toarray() : list 를 array로 바꿔준다.
print(y_train)
print(y_test)
print(y_train.shape)    # (60000, 10)
print(y_test.shape)     # (10000, 10)
'''



#2. Modling
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.layers import Reshape

model = Sequential()
# model.add(Conv2D(filters=16, kernel_size=(2,2), padding='same', strides=1, input_shape=(28,28,1)))
# model.add(MaxPooling2D(pool_size=2))
model.add(Dense(64, input_shape=(28, 28, 1)))   # Dense 그냥 3차원으로 넣어도 된다.
model.add(Dropout(0.5))
model.add(Flatten())  

model.add(Dense(64))  
model.add(Dense(784))   
model.add(Reshape((28, 28, 1)))   # 위 레이어의 shape를 (28, 28, 1) 로 바꿔준다.
model.add(Dense(1))

model.summary()
'''
# Compile, Train
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
# 체크포인트의 가중치를 저장할 파일경로 지정
modelpath='../data/modelcheckpoint/k57_mnist_{epoch:02d}-{val_loss:.4f}.hdf5'
es = EarlyStopping(monitor='val_loss', patience=10, mode='max')
cp = ModelCheckpoint(filepath=modelpath,monitor='val_loss', save_best_only=True, mode='auto')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, verbose=1)
model.compile(loss='mse',optimizer='adam',metrics=['accuracy'])
hist = model.fit(x_train, y_train, epochs=50, batch_size=16, validation_split=0.5, callbacks=[es, cp, reduce_lr])
# Evaluate, Predict
result = model.evaluate(x_test, y_test, batch_size=16)
print("loss : ", result[0])
print("accuracy : ", result[1])
y_pred = model.predict(x_test)
print(y_pred[0])
print(y_pred.shape)     # (10000, 28, 28, 1)
# 응용
# y_test 10개와 y_test 10개를 출력하시오
# print("y_test[:10] :\n", y_test[:10])
print("y_test[:10] :")
print(np.argmax(y_test[:10],axis=1))
y_predict = model.predict(x_test[:10])
print("y_pred[:10] :")  
print(np.argmax(y_predict,axis=1))
# 시각화
import matplotlib.pyplot as plt
plt.figure(figsize=(10,6))  # 판 사이즈 (가로 10, 세로 6)
plt.subplot(2, 1, 1)    # plot : 도화지 하나에 그림을 그리겠다.
                        # 2행 1열 중 첫 번째
                        # 만약 (3, 1, 1) 이라면 세 개의 plot이 있어야 한다. (3, 1, 1) (3, 1, 2) (3, 1, 3)
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')   # label=' ' >> legend에서 설정한 위치에 라벨이 표시된다.
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid()
# plt.title('손실비용') # 과제 : 한글 깨짐 오류 해결할 것
plt.title('Cost Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')   # loc 를 명시하지 않으면 그래프가 비어있는 지역에 자동으로 위치한다.
plt.subplot(2, 1, 2)    # 2행 1열 중 두 번째
# plt.plot(hist.history['accuracy'], marker='.', c='red', label='accuracy')   # label=' ' >> legend에서 설정한 위치에 라벨이 표시된다.
# plt.plot(hist.history['val_accuracy'], marker='.', c='blue', label='val_accuracy')
plt.plot(hist.history['accuracy'], marker='.', c='red')   
plt.plot(hist.history['val_accuracy'], marker='.', c='blue')
plt.grid()              # 모눈종이 격자위에 그리겠다.
# plt.title('정확도')   # 과제 : 한글 깨짐 오류 해결할 것
plt.title('Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
# plt.legend(loc='upper right')
plt.legend(['accuracy','val_accuracy'])
plt.show()
'''

# loss :  0.2846296727657318
# accuracy :  0.9150999784469604
# y_test[:10] :
# [7 2 1 0 4 1 4 9 5 9]
# y_pred[:10] :
# [7 2 1 0 4 1 4 9 5 9]

# ReduceLROnplateau
# loss :  0.22700025141239166
# accuracy :  0.9330000281333923
# y_test[:10] :
# [7 2 1 0 4 1 4 9 5 9]
# y_pred[:10] :
# [7 2 1 0 4 1 4 9 5 9]

# CNN -> DNN으로 변경
# loss :  0.2913188636302948
# accuracy :  0.9200000166893005
# y_test[:10] :
# [7 2 1 0 4 1 4 9 5 9]
# y_pred[:10] :
# [7 2 1 0 4 1 4 9 6 9]