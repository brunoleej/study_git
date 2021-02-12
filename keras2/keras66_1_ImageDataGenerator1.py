# ImageDataGenerator 
# Image Preprocessing
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ImageDataGenerator :  이미지 증폭시켜 더 많은 데이터로 훈련가능
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
test_datagen = ImageDataGenerator(rescale=1./255)   # 전처리만(test에서는 이미지 증폭을 할 필요 없음) 

#   1) flow : 이미지가 데이터화 되어 있을 때 사용함
#   2) flow_from_directory : 경로에 있는 파일을 데이터화
# 폴더 자체를 라벨링할 수 있다. --> (ex) ad : 0, noraml : 1

# train_generator
# ad / normal (앞 0, 뒤 1로 라벨링)
# x : (80, 150, 150, 1) -> 전부 0부터 1
# y : (80,) -> ad : 전부 0으로 라벨링, normal : 전부 1로 라벨링
xy_train = train_datagen.flow_from_directory(
    '../data/image/brain/train',    # 경로에 있는 이미지들을 데이터화 (80개 이미지 전부 0으로 라벨링)
    target_size=(150, 150),         # 이미지 크기
    # batch_size=5,                 # batch_size 만큼의 이미지가 생성(5장씩 이미지를 자름)
    # batch_size=10,                  
    # batch_size=160,               # 160으로 나누면 전체 데이터 나옴
    batch_size=200,                 # 160보다 높게 해도 최대 값 160이 나옴
    class_mode='binary'             # 이진분류 0, 1
)
# Result : Found 160 images belonging to 2 classes.

# test_generator
xy_test = test_datagen.flow_from_directory(
    '../data/image/brain/test',   
    target_size=(150, 150),        
    batch_size=5,                  
    # batch_size=10,               
    class_mode='binary'            
)
# Result : Found 120 images belonging to 2 classes.

# Data Check
print(xy_train)
# 결과 : <tensorflow.python.keras.preprocessing.image.DirectoryIterator object at 0x000001BCD5A68550>
# 데이터 경로 위치가 나옴

print(xy_train[0])

print(xy_train[0][0])        # x 만 출력됨
print(xy_train[0][0].shape)  # (10, 150, 150, 3)

print(xy_train[0][1])        # y 만 출력됨 [0. 1. 1. 1. 0. 0. 0. 1. 0. 1.]
print(xy_train[0][1].shape)  # (10,)
# print(xy_train[15][1].shape) # (10,) : 0 ~ 15까지 이미지 생성됨
