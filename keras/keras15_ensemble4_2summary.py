# ensemble (2 - 1 - 2)
# 모델 병합 : concatenate
# 모델 분기

# 실습
# 1 : 다 앙상블을 구현하시오 
# summary

import numpy as np

# Data
x1 = np.array( [range(100), range(301,401), range(1,101)] )         #(3, 100)
y1 = np.array( [range(711, 811), range(1, 101), range(201, 301)] )  

y2 = np.array([range(501, 601), range(711,811), range(100)])

x1 = np.transpose(x1)   #(100, 3)
y1 = np.transpose(y1)
y2 = np.transpose(y2)

from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test = train_test_split (x1, y1, shuffle=False, train_size=0.8)
y2_train, y2_test = train_test_split (y2, shuffle=False, train_size=0.8)

# Modeling
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

# Model 1
input1 = Input(shape=(3,))  # input : (3,)
dense1 = Dense(4, activation = 'relu')(input1)
dense1 = Dense(5, activation = 'relu')(dense1)

# 모델 분기 1 
output1 = Dense(20)(dense1)
output1 = Dense(20)(output1)
output1 = Dense(3)(output1) # y1 :output = 3

# 모델 분기 2
output2 = Dense(30)(dense1)
output2 = Dense(30)(output2)
output2 = Dense(3)(output2) # y2 :output = 3

# 모델 선언
model = Model(inputs = input1, outputs = [output1, output2])
model.summary()

'''
Model: "functional_1"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to
==================================================================================================
input_1 (InputLayer)            [(None, 3)]          0
__________________________________________________________________________________________________
dense (Dense)                   (None, 4)            16          input_1[0][0]
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 5)            25          dense[0][0]
__________________________________________________________________________________________________
dense_2 (Dense)                 (None, 20)           120         dense_1[0][0]
__________________________________________________________________________________________________
dense_5 (Dense)                 (None, 30)           180         dense_1[0][0]
__________________________________________________________________________________________________
dense_3 (Dense)                 (None, 20)           420         dense_2[0][0]
__________________________________________________________________________________________________
dense_6 (Dense)                 (None, 30)           930         dense_5[0][0]
__________________________________________________________________________________________________
dense_4 (Dense)                 (None, 3)            63          dense_3[0][0]
__________________________________________________________________________________________________
dense_7 (Dense)                 (None, 3)            93          dense_6[0][0]
==================================================================================================
Total params: 1,847
Trainable params: 1,847
Non-trainable params: 0
__________________________________________________________________________________________________
'''