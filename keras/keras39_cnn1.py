# Module import
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,Dense,Flatten,MaxPool2D

# Conv2D
# filters : layer에서 나갈 때 몇개의 filter를 만들 것인지
# kernel_size : filter(weight) size
# input_shape : batch_size, , RGB

# Model Declaration
model = Sequential([
    Conv2D(filters = 10,kernel_size = (2,2), strides = 1,
           padding = 'SAME',input_shape = (10,10,1)),  
    # MaxPool2D(pool_size=(2,3)),
    Conv2D(9,2,padding = 'VALID'),
    # Conv2D(9,(2,3))
    # Conv2D(8,(2,3)),
    Flatten(),
    Dense(1)
])
model.summary()

# RGB 'gray scale'(1)
'''
________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 9, 9, 10)          50        
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 8, 8, 9)           369       
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 7, 7, 8)           296       
_________________________________________________________________
flatten (Flatten)            (None, 392)               0         
_________________________________________________________________
dense (Dense)                (None, 1)                 393       
=================================================================
Total params: 1,108
Trainable params: 1,108
Non-trainable params: 0
'''

# Conv2D Parameter Calculate
# 1st Conv2D
# Conv2D => (None,9,9,10) => 이미지에 (2,2) filter를 10개 사용. 
# (2,2) 필터 1개에는 2 x 2 = 4개의 파라미터가 들어있습니다.
# 3-channel 각각에 서로 다른 파라미터들이 입력되므로 R, G, B에 해당하는 3이 곱해짐.
# Conv2D(filters = 10...) 에서의 10은 10개의 필터를 적용하여 다음층에서는 10개가 되도록 만든다는 뜻
# bias로 더해질 상수가 각각 채널마다 존재하므로 10개가 추가로 더해짐
# 2 x 2(필터 크기) x 1(입력 채널(RGB)) x 10(출력채널) + 10(출력 채널 bias) = 50

# 2nd Conv2D
# 


# RGB '3'
'''
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 9, 9, 10)          130       
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 8, 8, 9)           369       
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 7, 7, 8)           296       
_________________________________________________________________
flatten (Flatten)            (None, 392)               0         
_________________________________________________________________
dense (Dense)                (None, 1)                 393       
=================================================================
Total params: 1,188
Trainable params: 1,188
Non-trainable params: 0
'''

# Conv2D(9,(2,3))
'''
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 9, 9, 10)          130       
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 8, 8, 9)           369       
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 7, 6, 8)           440       
_________________________________________________________________
flatten (Flatten)            (None, 336)               0         
_________________________________________________________________
dense (Dense)                (None, 1)                 337       
=================================================================
Total params: 1,276
Trainable params: 1,276
Non-trainable params: 0
'''