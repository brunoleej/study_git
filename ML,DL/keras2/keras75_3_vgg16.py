# 전이학습 VGG16 (layer 총 16개)
# 당겨온 다음에 output shape을 바꿔본다.
# Layer 확인

from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential

vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

# 훈련을 시키지 않겠다.
vgg16.trainable = False             # False = 동결시킨다.
vgg16.summary()
print(len(vgg16.weights))             
print(len(vgg16.trainable_weights))   
'''
=================================================================
Total params: 14,714,688
Trainable params: 0
Non-trainable params: 14,714,688
_________________________________________________________________
26
0
'''

# Sequential에 vgg16을 넣어준다.
model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(1))#, activation='softmax'))

model.summary()
print("전체 가중치의 수 : ", len(model.weights))             
print("동결한 후, 훈련되는 가중치의 수 : ", len(model.trainable_weights)) 
'''
=================================================================
Total params: 14,719,879
Trainable params: 5,191
Non-trainable params: 14,714,688
_________________________________________________________________
32  # 26 -> 32 : vgg16에서 레이어 3개 추가된 것
6
'''

### Layer 확인 ###

import pandas as pd 
pd.set_option('max_colwidth', -1)   # set_option('max_colwidth', -1) : 출력할 컬럼의 길이 지정
layers = [(layer, layer.name, layer.trainable) for layer in model.layers]           # layer, layer.name, layer.trainable 값을 layers로 반환한다.
aaa = pd.DataFrame(layers, columns=['Layer Type', 'Layer Name', 'Layer Trainable'])

print(aaa)
'''
                                                                            Layer Type Layer Name  Layer Trainable
0  <tensorflow.python.keras.engine.functional.Functional object at 0x0000023B0426F550>  vgg16      False
1  <tensorflow.python.keras.layers.core.Flatten object at 0x0000023B0427FB20>           flatten    True
2  <tensorflow.python.keras.layers.core.Dense object at 0x0000023B042A4430>             dense      True
3  <tensorflow.python.keras.layers.core.Dense object at 0x0000023B042B8370>             dense_1    True
4  <tensorflow.python.keras.layers.core.Dense object at 0x0000023B042C2EE0>             dense_2    True
'''