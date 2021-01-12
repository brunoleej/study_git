# '행'이 다른 앙상블 모델에 대해 공부하라 
# 결론 >> 앙상블을 사용할 때 무조건 행의 크기를 맞춰줘야 한다. 
# 데이터를 삭제하려면 행 전체를 다 날려야 한다. 
    # (혹은 이상치를 이상적으로 수정한다.) 
    # (혹은 비워있는 부분을 채운다.) - 위, 아래 있는 값을 참고해서 채운다. / 모델을 돌려서 나온 y=ax+b를 기준으로 비워있는 값을 채운다. 
import numpy as np

#1. DATA
x1 = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
                [5,6,7],[6,7,8],[7,8,9],[8,9,10],
                [9,10,11],[10,11,12]])
x2 = np.array([[10,11,12],[20,30,40],[30,40,50],[40,50,60],
                [50,60,70],[60,70,80],[70,80,90],[80,90,100],
                [90,100,110],[100,110,120],
                [2,3,4],[3,4,5],[4,5,6]])
y1 = np.array([4,5,6,7,8,9,10,11,12,13]) 
y2 = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70]) 
x1_predict = np.array([55,65,75])
x2_predict = np.array([65,75,85])

# print(x1.shape)            # (10, 3)
# print(x2.shape)            # (13, 3)
# print(y1.shape)            # (10, )
# print(y2.shape)            # (13, )
# print(x1_predict.shape)    # (3,) -> Dense (1, 3) -> LSTM (1, 3, 1)

x1_predict = x1_predict.reshape(1, 3)
x2_predict = x2_predict.reshape(1, 3)

# preprocessing

from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, train_size=0.9, shuffle=True, random_state=44)  
x2_train, x2_test, y2_train, y2_test = train_test_split(x2, y2, train_size=0.9, shuffle=True, random_state=44)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x1_train)
scaler.fit(x2_train)
x1_train = scaler.transform(x1_train)
x2_train = scaler.transform(x2_train)
x1_test = scaler.transform(x1_test)
x2_test = scaler.transform(x2_test)
x1_predict = scaler.transform(x1_predict)
x2_predict = scaler.transform(x2_predict)

print(x1_train.shape)             # (9, 3)
print(x2_train.shape)             # (11, 3)
print(x1_test.shape)              # (1, 3)
print(x2_test.shape)              # (2, 3)

print(y1_test.shape)              # (1, )
print(y2_test.shape)              # (2, )

# *** LSTM 모델을 사용하기 위해서 데이터를 3차원으로 만들어준다. ***
x1_train = x1_train.reshape(9, 3, 1)
x2_train = x2_train.reshape(11, 3, 1)

x1_test = x1_test.reshape(1, 3, 1)
x2_test = x2_test.reshape(2, 3, 1)

x1_predict = x1_predict.reshape(1, 3, 1)
x2_predict = x2_predict.reshape(1, 3, 1)

#2. Modeling
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, LSTM

# 모델 구성
# model 1
input1 = Input(shape=(3,1))
dense1 = LSTM(65, activation='relu')(input1)
dense1 = Dense(13, activation='relu')(dense1)
dense1 = Dense(13, activation='relu')(dense1)

# model2
input2 = Input(shape=(3,1))
dense2 = LSTM(65, activation='relu')(input2)
dense2 = Dense(13, activation='relu')(dense2)
dense2 = Dense(13, activation='relu')(dense2)

# concatenate
from tensorflow.keras.layers import concatenate
merge1 = concatenate([dense1, dense2]) 
middle1 = Dense(39)(merge1)
middle1 = Dense(13)(middle1)

# 모델 분기
# 분기 1
output1 = Dense(13)(middle1)
output1 = Dense(1)(output1) # 최종 output = 1

# 분기l 2
output2 = Dense(13)(middle1)
output2 = Dense(1)(output1) # 최종 output = 1

# 모델 선언
model = Model(inputs=[input1, input2], outputs=[output1, output2])

model.summary()

#3. Compile, Train
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

model.fit([x1_train,x2_train], [y1_train, y2_train], epochs=12, batch_size=5, validation_split=0.1)

#4. Evaluate, Predict
loss = model.evaluate([x1_test,x2_test], [y1_test, y2_test], batch_size=1)
print("loss, mae : ", loss)

'''
ValueError: Data cardinality is ambiguous:
  x sizes: 1, 2
  y sizes: 1, 2
Please provide data which shares the same first dimension.
'''


y_pred1, y_pred2 = model.predict([x1_predict, x2_predict])

print("y_pred : ", y_pred1, y_pred2)