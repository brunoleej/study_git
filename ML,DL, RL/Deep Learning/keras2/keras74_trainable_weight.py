# weight 값 확인

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. DATA
x = np.array([1,2,3,4,5])
y = np.array([1,2,3,4,5])

#2. Model
model = Sequential()
model.add(Dense(4, input_dim=1))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))

model.summary()
'''
Layer (type)                 Output Shape              Param #
=================================================================
dense (Dense)                (None, 4)                 8
_________________________________________________________________
dense_1 (Dense)              (None, 3)                 15
_________________________________________________________________
dense_2 (Dense)              (None, 2)                 8
_________________________________________________________________
dense_3 (Dense)              (None, 1)                 3
=================================================================
Total params: 34
Trainable params: 34
Non-trainable params: 0
_________________________________________________________________
'''

print(model.weights)
# 레이어 별로 weight 값을 알 수 있다.
'''
[<tf.Variable 'dense/kernel:0' shape=(1, 4) dtype=float32, numpy=

# model.add(Dense(4, input_dim=1)) 에 해당하는 weight 4개 
array([[-0.9674869 , -0.28564394, -1.0481684 ,  0.58860576]],
      dtype=float32)>, <tf.Variable 'dense/bias:0' shape=(4,) dtype=float32, numpy=array([0., 0., 0., 0.], dtype=float32)>, <tf.Variable 'dense_1/kernel:0' shape=(4, 3) dtype=float32, numpy=        

# model.add(Dense(3)) 에 해당하는 weight 12개
array([[ 0.75796044,  0.34799993, -0.17284566],
       [ 0.56958425,  0.4618795 , -0.5898894 ],
       [-0.18093926, -0.6530194 , -0.62394094],
       [-0.4815015 ,  0.19216633, -0.23627716]], dtype=float32)>, <tf.Variable 'dense_1/bias:0' shape=(3,) dtype=float32, numpy=array([0., 0., 0.], dtype=float32)>, <tf.Variable 'dense_2/kernel:0' shape=(3, 2) dtype=float32, numpy=

# model.add(Dense(2)) 에 해당하는 weight 6개
array([[ 0.03967738, -0.9099102 ],
       [-0.9870162 ,  0.18094039],
       [ 0.17763889, -0.56159616]], dtype=float32)>, <tf.Variable 'dense_2/bias:0' shape=(2,) dtype=float32, numpy=array([0., 0.], dtype=float32)>, <tf.Variable 'dense_3/kernel:0' shape=(2, 1) dtype=float32, numpy=

# model.add(Dense(1)) 에 해당하는 weight 2개
array([[-1.0873432],
       [ 0.8842114]], dtype=float32)>, <tf.Variable 'dense_3/bias:0' shape=(1,) dtype=float32, numpy=array([0.], dtype=float32)>]
'''

print(model.trainable_weights)
# 훈련시키는 weight 값들
# 전이학습을 시킬 때, 계산하지 않아도 된는 weight값들이 생긴다.
'''
[<tf.Variable 'dense/kernel:0' shape=(1, 4) dtype=float32, numpy=

array([[ 1.0290966 , -0.35784793, -0.44488513, -0.54471827]],
      dtype=float32)>, <tf.Variable 'dense/bias:0' shape=(4,) dtype=float32, numpy=array([0., 0., 0., 0.], dtype=float32)>, <tf.Variable 'dense_1/kernel:0' shape=(4, 3) dtype=float32, numpy=

array([[-0.24531418, -0.4465423 , -0.6305989 ],
       [ 0.11009729,  0.17650831, -0.02232051],
       [-0.5751972 , -0.07005769,  0.25724435],
       [ 0.20997024,  0.87521803, -0.07706308]], dtype=float32)>, <tf.Variable 'dense_1/bias:0' shape=(3,) dtype=float32, numpy=array([0., 0., 0.], dtype=float32)>, <tf.Variable 'dense_2/kernel:0' shape=(3, 2) dtype=float32, numpy=      

array([[-0.5478213 ,  0.44611812],
       [ 0.5152118 ,  0.3553182 ],
       [ 0.37134647, -0.821482  ]], dtype=float32)>, <tf.Variable 'dense_2/bias:0' shape=(2,) dtype=float32, numpy=array([0., 0.], dtype=float32)>, <tf.Variable 'dense_3/kernel:0' shape=(2, 1) dtype=float32, numpy=

array([[-0.55704105],
       [ 0.06127   ]], dtype=float32)>, <tf.Variable 'dense_3/bias:0' shape=(1,) dtype=float32, numpy=array([0.], dtype=float32)>]
'''

print(len(model.weights))           # 8개   -   각 레이어 당 weight 하나 bias 하나씩 계산한다.
print(len(model.trainable_weights)) # 8개   -   각 레이어 당 weight 하나 bias 하나씩 계산한다.
