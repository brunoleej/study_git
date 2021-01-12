# '열'이 다른 앙상블 모델에 대해 공부하라 
# 결론 >> 앙상블을 사용할 때 열이 달라도 된다.

import numpy as np

#1. DATA
x1 = np.array([[1,2],[2,3],[3,4],[4,5],
                [5,6],[6,7],[7,8],[8,9],
                [9,10],[10,11],
                [20,30],[30,40],[40,50]])
x2 = np.array([[10,11,12],[20,30,40],[30,40,50],[40,50,60],
                [50,60,70],[60,70,80],[70,80,90],[80,90,100],
                [90,100,110],[100,110,120],
                [2,3,4],[3,4,5],[4,5,6]])

y1 = np.array([[10,11,12],[20,30,40],[30,40,50],[40,50,60],
                [50,60,70],[60,70,80],[70,80,90],[80,90,100],
                [90,100,110],[100,110,120],
                [2,3,4],[3,4,5],[4,5,6]])
y2 = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70]) 

# Data
x1_predict = np.array([55,65])    # >> 결과 3개
x2_predict = np.array([65,75,85]) # >> 결과 1개

print(x1.shape)            # (13, 2)
print(x2.shape)            # (13, 3)
print(y1.shape)            # (13, 3)
print(y2.shape)            # (13, )
print(x1_predict.shape)    # (2,) 
print(x2_predict.shape)    # (3,) 

x1_predict = x1_predict.reshape(1, 2)
x2_predict = x2_predict.reshape(1, 3)

x1 = x1.reshape(13, 2, 1)
x2 = x2.reshape(13, 3, 1)

#2. Modeling
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, LSTM

# 모델 구성
# model 1
input1 = Input(shape=(2,1))
dense1 = LSTM(65, activation='relu')(input1)
dense1 = Dense(13, activation='relu')(dense1)
dense1 = Dense(13, activation='relu')(dense1)

# model2
input2 = Input(shape=(3,1))
dense2 = LSTM(65, activation='relu')(input2)
dense2 = Dense(13, activation='relu')(dense2)
dense2 = Dense(13, activation='relu')(dense2)

# 모델 병합 : concatenate
from tensorflow.keras.layers import concatenate
merge1 = concatenate([dense1, dense2]) 
middle1 = Dense(39)(merge1)
middle1 = Dense(13)(middle1)

# 모델 분기
# 분기 1
output1 = Dense(13)(middle1)
output1 = Dense(3)(output1) # 최종 output = 1

# 분기l 2
output2 = Dense(13)(middle1)
output2 = Dense(1)(output2) # 최종 output = 1

# 모델 선언
model = Model(inputs=[input1, input2], outputs=[output1, output2])

model.summary()

#3. Compile, Train
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit([x1, x2], [y1, y2], epochs=12, batch_size=5, validation_split=0.1)

#4. Evaluate, Predict
# loss = model.evaluate([x1_test,x2_test], [y1_test, y2_test], batch_size=1)
loss = model.evaluate([x1,x2], [y1, y2], batch_size=1)
print("loss, mae : ", loss)

'''
ValueError: Data cardinality is ambiguous:
  x sizes: 1, 2
  y sizes: 1, 2
Please provide data which shares the same first dimension.
'''


y_pred1, y_pred2 = model.predict([x1_predict, x2_predict])

print("y_pred : ", y_pred1, y_pred2)

# loss, mae :  [3195.98681640625, 2573.2412109375, 622.7456665039062, 40.8387451171875, 13.721480369567871]
# y_pred :  [[-0.6377039 16.92513   25.793715 ]] [[24.777428]]