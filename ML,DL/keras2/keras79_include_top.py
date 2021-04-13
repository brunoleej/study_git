from tensorflow.keras.applications import VGG16

model = VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3))   # fc1 부분 이하를 cut시킬 수 있다. 하단에 내가 원하는 사이즈를 커스텀할 수 있다.
# model = VGG16(weights='imagenet', include_top=True, input_shape=(224,224,3))  # fc1 이하 dense 까지 포함됨
# model = VGG16()   # include_top=True 가 디폴트임

model.trainable = False
model.summary()
print(len(model.weights))
print(len(model.trainable_weights))

'''
# VGG16  (include_top = False)
=================================================================
Total params: 14,714,688
Trainable params: 0
Non-trainable params: 14,714,688
_________________________________________________________________
26
0

# VGG16  (include_top = True)
=================================================================
Total params: 138,357,544
Trainable params: 0
Non-trainable params: 138,357,544
_________________________________________________________________
32
0

# model=VGG16() : (include_top = True)와 동일하다.
=================================================================
Total params: 138,357,544
Trainable params: 0
Non-trainable params: 138,357,544
_________________________________________________________________
32
0
'''
