# ImageDataGenerator 사용법 
# 이미지 전처리
# npy 저장 & load

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# [1] 선언
# ImageDataGenerator 선언 >> 이미지를 증폭시킴으로써 더 많은 데이터로 훈련시킬 수 있다.
train_datagen = ImageDataGenerator(
    rescale=1./255,         # 전처리
    horizontal_flip=True,   # 수평
    vertical_flip=True,     # 수직
    width_shift_range=0.1,  # 좌우
    height_shift_range=0.1, # 상하
    rotation_range=5,       # 회전
    zoom_range=1.2,         # 확대
    shear_range=0.7,        # 밀림 정도
    fill_mode='nearest'     # 빈 자리를 유사한 값으로 채움
)
test_datagen = ImageDataGenerator(rescale=1./255)   # 전처리만 한다. >> test에서는 이미지 증폭을 할 필요가 없다.

#   1) flow >> 이미지가 데이터화 되어 있을 때 사용함
#   2) flow_from_directory >> 폴더 안에 있는 파일을 데이터화 한다.
# 폴더 자체를 라벨링할 수 있다. --> (ex) ad : 0, noraml : 1

# train_generator
# ad / normal (앞에 있는 걸 0, 뒤에 있는 걸 1로 라벨링됨)
# x : (80, 150, 150, 1) >> 모두 0부터 1
# y : (80,)             >> ad : 모두 0으로 라벨링 / normal : 모두 1로 라벨링
xy_train = train_datagen.flow_from_directory(
    '../data/image/brain/train',    # 경로 설정 > 해당 폴더에 있는 이미지들을 데이터화 하겠다. > 80개 이미지 전부 0으로 라벨링
    target_size=(150, 150),         # 이미지 크기
    batch_size=160,                 # 160장으로 나누면 전체 데이터가 나온다.
    class_mode='binary'             # 이진분류 0, 1
)
# 결과 : Found 160 images belonging to 2 classes.

#   [참고]
#   * fit_generator 는 배치사이즈 신경 안써도 된다. 자동으로 연산해준다.
#   * fit 할 때는 배치사이즈 고려해서 잘라줘야 한다.

# test_generator
xy_test = test_datagen.flow_from_directory(
    '../data/image/brain/test',    # 경로 설정 > 해당 폴더에 있는 이미지들을 데이터화 하겠다. > 80개 이미지 전부 0으로 라벨링
    target_size=(150, 150),        # 이미지 크기
    batch_size=120,                 # 120장으로 나누면 전체 데이터가 나온다.
    class_mode='binary'            # 이진분류 0, 1
)
# 결과 : Found 120 images belonging to 2 classes.

# 데이터 확인
print(xy_train)
# 결과 : <tensorflow.python.keras.preprocessing.image.DirectoryIterator object at 0x000001BCD5A68550>
# 데이터를 한 군데에 모아두었다는 뜻

print(xy_train[0])
# 160장을 batch_size=10으로 나누었음 >> 0부터 15까지 있음
# 0번째의 0번째 : x
# 0번째의 1번째 : y

print(xy_train[0][0])        # >> x 만 출력된다.
print(xy_train[0][0].shape)  # (160, 150, 150, 3)
print("==========================================")
print(xy_train[0][1])        # >> y 만 출력된다. [0. 1. 1. 1. 0. 0. 0. 1. 0. 1.]
print(xy_train[0][1].shape)  # (120,)
# print(xy_train[15][1].shape) # (10,) >> 0부터 15까지 이미지가 생성된다.
print("==========================================")
# npy 저장
# np.save('../data/image/brain/npy/keras66_train_x.npy', arr=xy_train[0][0]) # train_x 가 저장된다.
# np.save('../data/image/brain/npy/keras66_train_y.npy', arr=xy_train[0][1]) # train_y 가 저장된다.
# np.save('../data/image/brain/npy/keras66_test_x.npy', arr=xy_test[0][0]) # test_x 가 저장된다.
# np.save('../data/image/brain/npy/keras66_test_y.npy', arr=xy_test[0][1]) # test_y 가 저장된다.

# npy load
x_train = np.load('../data/image/brain/npy/keras66_train_x.npy')
x_test = np.load('../data/image/brain/npy/keras66_test_x.npy')
y_train = np.load('../data/image/brain/npy/keras66_train_y.npy')
y_test = np.load('../data/image/brain/npy/keras66_test_y.npy')

print(x_train.shape, x_test.shape)  # (160, 150, 150, 3) (120, 150, 150, 3)
print(y_train.shape, y_test.shape)  # (160,) (120,)
