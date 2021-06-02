# ensemble ( 2 - 1 - 1)
# 모델 병합 : concatenate
# 실습 
# 다 : 1 앙상블을 구현하시오 (y2 제거, 분기하는 부분을 뺀다.)
# summary
import numpy as np

# Data
x1 = np.array([range(100), range(301,401), range(1,101)])         #(3, 100)
x2 = np.array([range(101, 201), range(411,511),range(100,200)])

y1 = np.array( [range(711, 811), range(1, 101), range(201, 301)] )  

x1 = np.transpose(x1)   #(100, 3)
x2 = np.transpose(x2)
y1 = np.transpose(y1)

from sklearn.model_selection import train_test_split
x1_train, x1_test, x2_train, x2_test, y1_train, y1_test = train_test_split (x1, x2, y1, shuffle=False, train_size=0.8)

print(x1_train.shape)   #(80, 3)
print(x2_test.shape)    #(20, 3)
print(y1_train.shape)   #(80, 3)

# Modeling
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

# 두 모델을 합쳤다가 다시 분리하는 과정
# Model 1
input1 = Input(shape=(3,)) #input = 3
dense1 = Dense(4, activation = 'relu')(input1)
dense1 = Dense(5, activation = 'relu')(dense1)

# Model 2
input2 = Input(shape=(3,))  #input = 3
dense2 = Dense(6, activation = 'relu')(input2)
dense2 = Dense(7, activation = 'relu')(dense2)

# 모델 병합 : concatenate
from tensorflow.keras.layers import concatenate, Concatenate

# merge도 layers에 속해있으므로 layer를 구성한다.
merge1 = concatenate([dense1, dense2]) # 두 모델의 마지막 층에 있는 레이어를 합친다.
middle1 = Dense(10)(merge1)
middle1 = Dense(10)(merge1)


# merge의 마지막 층을 가져온다. (둘로 나누지 않는다.)
# 모델 분기 1 
output1 = Dense(20)(middle1)
output1 = Dense(3)(output1) # y1 :output = 3 (마지막 아웃푸ㅛ)

# 모델 선언
# 최종적인 input, output을 넣어서 모델 구성
# 두 개 이상은 리스트로 묶는다.
model = Model(inputs = [input1, input2], outputs = output1)
model.summary()
'''
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
dense_5 (Dense)                 (None, 10)           130         concatenate[0][0]
__________________________________________________________________________________________________
dense_6 (Dense)                 (None, 20)           220         dense_5[0][0]
__________________________________________________________________________________________________
dense_7 (Dense)                 (None, 3)            63          dense_6[0][0]
==================================================================================================
Total params: 527
Trainable params: 527
Non-trainable params: 0
__________________________________________________________________________________________________
'''