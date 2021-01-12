# ensemble (2 - 1 - 2)
# 모델 병합 : concatenate
# 모델 분기
# summary 분석


import numpy as np

# Data
x1 = np.array( [range(100), range(301,401), range(1,101)] )         #(3, 100)
y1 = np.array( [range(711, 811), range(1, 101), range(201, 301)] )  

x2 = np.array([range(101, 201), range(411,511),range(100,200)])
y2 = np.array([range(501, 601), range(711,811), range(100)])

x1 = np.transpose(x1)   #(100, 3)
x2 = np.transpose(x2)
y1 = np.transpose(y1)
y2 = np.transpose(y2)

from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test = train_test_split (x1, y1, shuffle=False, train_size=0.8)
x2_train, x2_test, y2_train, y2_test = train_test_split (x2, y2, shuffle=False, train_size=0.8)

# Modeling
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

# 두 모델을 합쳤다가 다시 분리하는 과정

# Model 1 (3-4-5)
input1 = Input(shape=(3,)) #input = 3
dense1 = Dense(4, activation = 'relu')(input1)
dense1 = Dense(5, activation = 'relu')(dense1)

# Model 2 (3-6-7-8-9)
input2 = Input(shape=(3,))  #input = 3
dense2 = Dense(6, activation = 'relu')(input2)
dense2 = Dense(7, activation = 'relu')(dense2)

# 모델 병합 : concatenate (10-10)
from tensorflow.keras.layers import concatenate, Concatenate

merge1 = concatenate([dense1, dense2]) # 두 모델의 마지막 층에 있는 레이어를 합친다.
middle1 = Dense(10)(merge1)
middle1 = Dense(10)(middle1)

# 둘로 합쳤던 것을 다시 나눈다. merge의 마지막 층을 가져온다.
# 모델 분기 1 (20-20-3)
output1 = Dense(20)(middle1)
output1 = Dense(20)(output1)
output1 = Dense(3)(output1) # y1 :output = 3

# 모델 분기 2 (30-30-30-3)
output2 = Dense(30)(middle1)
output2 = Dense(30)(output2)
output2 = Dense(30)(output2)
output2 = Dense(3)(output2) # y2 :output = 3

# 모델 선언 
model = Model(inputs = [input1, input2], outputs = [output1, output2])
model.summary()

"""
Model: "functional_1"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to
==================================================================================================
input_1 (InputLayer)            [(None, 3)]          0
__________________________________________________________________________________________________
input_2 (InputLayer)            [(None, 3)]          0
__________________________________________________________________________________________________
dense (Dense)                   (None, 4)            16          input_1[0][0]
__________________________________________________________________________________________________
dense_2 (Dense)                 (None, 6)            24          input_2[0][0]
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 5)            25          dense[0][0]
__________________________________________________________________________________________________
dense_3 (Dense)                 (None, 7)            49          dense_2[0][0]
__________________________________________________________________________________________________
concatenate (Concatenate)       (None, 12)           0           dense_1[0][0]
                                                                 dense_3[0][0]
__________________________________________________________________________________________________
dense_4 (Dense)                 (None, 10)           130         concatenate[0][0]
__________________________________________________________________________________________________
dense_5 (Dense)                 (None, 10)           110         dense_4[0][0]
__________________________________________________________________________________________________
dense_9 (Dense)                 (None, 30)           330         dense_5[0][0]
__________________________________________________________________________________________________
dense_6 (Dense)                 (None, 20)           220         dense_5[0][0]
__________________________________________________________________________________________________
dense_10 (Dense)                (None, 30)           930         dense_9[0][0]
__________________________________________________________________________________________________
dense_7 (Dense)                 (None, 20)           420         dense_6[0][0]
__________________________________________________________________________________________________
dense_11 (Dense)                (None, 30)           930         dense_10[0][0]
__________________________________________________________________________________________________
dense_8 (Dense)                 (None, 3)            63          dense_7[0][0]
__________________________________________________________________________________________________
dense_12 (Dense)                (None, 3)            93          dense_11[0][0]
==================================================================================================
Total params: 3,340
Trainable params: 3,340
Non-trainable params: 0
__________________________________________________________________________________________________

"""
