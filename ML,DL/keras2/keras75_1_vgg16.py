# 전이학습 VGG16 (layer 총 16개)

from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential

# vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
# weights='imagenet' : 가중치를 가져온다.
# include_top=False  : 내가 원하는 사이즈로 바꿔서 넣을 수 있다.
# include_top=True   : 'imagenet'에 있는 사이즈를 가져온다.

model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

# 훈련을 시키지 않겠다.
model.trainable = False
model.summary()
print(len(model.weights))             
print(len(model.trainable_weights))   
'''
=================================================================
Total params: 14,714,688
Trainable params: 0
Non-trainable params: 14,714,688
_________________________________________________________________
len(model.weights) 26
len(model.trainable_weights) 0
'''

# 훈련을 시키겠다.
model.trainable = True
model.summary()
print(len(model.weights))            
print(len(model.trainable_weights))   
'''
=================================================================
Total params: 14,714,688
Trainable params: 14,714,688
Non-trainable params: 0
_________________________________________________________________
len(model.weights) 26
len(model.trainable_weights) 26
'''