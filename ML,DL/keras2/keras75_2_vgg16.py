# 전이학습 VGG16 (layer 총 16개)
# 당겨온 다음에 output shape을 바꿔본다.

from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential

vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

# 훈련을 시키지 않겠다.
vgg16.trainable = False             # = 동결시킨다.
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