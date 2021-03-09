import numpy as np

from sklearn.datasets import load_boston

dataset = load_boston()

#1. DATA

# x = np.array(range(1,100))
# y = np.array(range(101,200))

# from sklearn.model_selection import train_test_split
# x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,shuffle = True)
# x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.8, shuffle = True)

# x1 = np.array( [range(100), range(1, 101), range(101,201)] ) 
# y1 = np.array( [range(511,611), range(611,711), range(711,811)] )  

# x2 = np.array( [range(100), range(1, 101), range(101,201)] ) 
# y2 = np.array( [range(511,611), range(611,711), range(711,811)] )  

# x1 = np.transpose(x1) 
# y1 = np.transpose(y1) 
# x2 = np.transpose(x2) 
# y2 = np.transpose(y2) 

x = dataset.data
y = dataset.target

# x = x/711.


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split\
    (x,y,train_size=0.8, shuffle=True,random_state=55)
# x2_train, x2_test, y2_train, y2_test = train_test_split\
#     (x2,y2,train_size=0.8, shuffle=True,random_state=55)
x_train, x_val, y_train, y_val = train_test_split\
    (x_train,y_train,train_size=0.8, shuffle=True,random_state=55)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)

#2. Modeling
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

# model = Sequential()
# model.add(Dense(10, input_shape=(2,), activation='relu'))
# model.add(Dense(10))
# model.add(Dense(10))
# model.add(Dense(10))
# model.add(Dense(2))

# Model1
input1 = Input(shape=(13,))
dense1 = Dense(10,activation='relu')(input1)
dense1 = Dense(10)(dense1)
dense1 = Dense(10)(dense1)
outputs = Dense(1)(dense1)

# Model2
# input2 = Input(shape=(3,))
# dense2 = Dense(10,activation='relu')(input2)
# dense2 = Dense(10)(dense2)
# dense2 = Dense(10)(dense2)

# 모델 병합
# from tensorflow.keras.layers import concatenate

# merge1 = concatenate([dense1,dense2])
# middle1 = Dense(10)(merge1)
# middle1 = Dense(10)(middle1)
# middle1 = Dense(10)(middle1)

# 모델 분기
# 1
# output1 = Dense(10)(dense1)
# output1 = Dense(10)(output1)
# output1 = Dense(13)(output1)

# 2
# output2 = Dense(10)(dense1)
# output2 = Dense(10)(output2)
# output2 = Dense(13)(output2)

model = Model(inputs = input1, outputs = outputs)

model.summary()

#3. Cpmpile, Train
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

from tensorflow.keras.callbacks import EarlyStopping
earlystopping = EarlyStopping(monitor='loss',mode='min',patience=5) 
model.fit(x_train, x_train,epochs=10,batch_size=1, \
    validation_data=(x_val, y_val), verbose=1,callbacks=[earlystopping])

#4. Evalute, Predict
loss = model.evaluate(x_test,y_test,batch_size=1)
print("loss : ", loss)

y_predict = model.predict(x_test)
print("y1_predict, y2_predict : ", y_predict)

# RMSE
from sklearn.metrics import mean_squared_error
def RMSE (y_test, y_predict) :
    return np.sqrt(mean_squared_error(y_test, y_predict))
RMSE = RMSE(y_test, y_predict)
print("RMSE : ", RMSE)

# R2
from sklearn.metrics import r2_score
R2 = r2_score(y_test, y_predict)
print("R2 : ", R2)