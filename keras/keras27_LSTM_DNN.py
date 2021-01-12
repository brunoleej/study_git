# keras23_LSTM3_scale 을 DNN으로 코딩
# 결과치 비교

# DNN으로 23번 파일보다 loss를 좋게 만들것
import numpy as np

# 1. Data
x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
              [5,6,7],[6,7,8],[7,8,9],[8,9,10],
              [9,10,11],[10,11,12],
              [20,30,40],[30,40,50],[40,50,60]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x_pred = np.array([50,60,70])

# x_pred.reshape
x_pred = x_pred.reshape(1,3)

# LSTM
# result => 80
print(x.shape,y.shape) # (13, 3) (13,)
# x = x.reshape((13,3,1)).astype('float32')
print(x)
print(x.shape)

# train_test_split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3, random_state=1)

# preprocessing
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_pred = scaler.transform(x_pred)

# Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

model = Sequential([
    Dense(10,activation = 'relu',input_shape=(3,)),
    Dense(30),
    Dense(20),
    Dense(10,activation = 'relu'),
    Dense(5),
    Dense(1)
])
# print(model.summary())

# Earlystopping
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor = 'loss',patience=20,mode = 'auto')

# Compile, Train
model.compile(loss = 'mse',optimizer = 'adam',metrics=['mae'])
model.fit(x_train,y_train,epochs = 500,batch_size = 1,validation_split = 0.2,callbacks=[early_stopping])



# evaluate, predict
loss = model.evaluate(x_test,y_test)
print('loss: ',loss)

y_pred = model.predict(x_pred)
print('y_pred: ',y_pred)

# loss:  [26.84933853149414, 2.747281312942505]
# y_pred:  [[85.07508]]
