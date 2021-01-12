# 중요함
# model.summary() 에 대하여
# parameter 개수에 대하여

import numpy as np
import tensorflow as tf 

# Data
x = np.array([1,2,3])
y = np.array([1,2,3])

# Modeling
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense 

model = Sequential([    # 순차모델 구성
    Dense(32,activation = 'linear',input_dim = 1),   # input : 1, output : 5, 'linear'
    Dense(16,name = 'su_layer1'),
    Dense(16,name = 'su_layer2'),
    Dense(8, name = 'su_layer1'),   # ValueError: All layers added to a Sequential model should have unique names.
    Dense(1)    # output : 1
])

model.summary()

'''
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #  (연산의 개수)
=================================================================
dense (Dense)                (None, 5)                 10
                             (행무시, 5)                          
_________________________________________________________________
dense_1 (Dense)              (None, 3)                 18
_________________________________________________________________
dense_2 (Dense)              (None, 4)                 16
_________________________________________________________________
dense_3 (Dense)              (None, 1)                 5
=================================================================
Total params: 49  (<--연산의 총합)
Trainable params: 49
Non-trainable params: 0
_________________________________________________________________
* parameter 개수 : bias를 node의 개수에 포함시켜서 다음 layer의 node 개수와 곱한다.
(bias는 모든 파라미터 연산에 포함되기 때문에 노드 개수에 +1 해서 계산하면 됨)
'''

# 실습2 + 과제
# ensemble1, 2, 3, 4에 대해 summary를 계산하고 이해한 것을 과제로 제출할 것
# ex) 모델1과 모델2를 왔다갔다 함

# layer를 만들 때 'name' 에 대해 확인하고 설명할 것 (레이어의 이름)
# layer name을 알아야 하는 이유를 찾아라/ layer 를 반드시 써야할 때가 언제인지 말해라/-이름이 충돌되는 경우, 어떨 때 충돌이 되는지? 찾아라
