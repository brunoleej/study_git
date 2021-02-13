# ImageDataGenerator 사용법 
# 이미지 전처리
# npy 저장 & load

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPool2D, BatchNormalization, AveragePooling2D
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

#1. DATA
# npy load
x_train = np.load('../data/image/brain/npy/keras66_train_x.npy')
x_test = np.load('../data/image/brain/npy/keras66_test_x.npy')
y_train = np.load('../data/image/brain/npy/keras66_train_y.npy')
y_test = np.load('../data/image/brain/npy/keras66_test_y.npy')

print(x_train.shape, x_test.shape)  # (160, 150, 150, 3) (120, 150, 150, 3)
print(y_train.shape, y_test.shape)  # (160,)             (120,)

x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, train_size=0.8, random_state=47)
print(x_train.shape, x_test.shape, x_valid.shape)  # (128, 150, 150, 3) (120, 150, 150, 3) (32, 150, 150, 3)
print(y_train.shape, y_test.shape, y_valid.shape)  # (128,) (120,) (32,)

#2. Modeling
model = Sequential()
model.add(Conv2D(32, (2,2), padding='same', activation='relu', input_shape=(150, 150, 3)))
model.add(BatchNormalization())
model.add(Conv2D(32, (2,2), padding='same', activation='relu'))
model.add(BatchNormalization())

model.add(Conv2D(64, (2,2), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(64, (2,2), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Conv2D(128, (2,2), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(128, (2,2), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(32, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(1, activation='sigmoid'))

#3. Compile, Train

es = EarlyStopping(monitor='val_loss', patience=20, mode='min')
lr = ReduceLROnPlateau(monitor ='val_loss', factor=0.3, patience = 10, mode='min')

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=100, batch_size=16, validation_data=(x_valid, y_valid) , callbacks=[es, lr])

#4. Evaluate, Predict
loss, acc = model.evaluate(x_test, y_test, batch_size=16)
print("loss : ", loss)
print("acc : ", acc)

# loss :  0.5336507558822632
# acc :  0.7416666746139526

# y_pred = model.predict(x_test)

# print("y_pred : \n", y_pred)
# print(y_pred.shape) # (120, 1)
# print(np.argmax(y_pred,axis=1))
