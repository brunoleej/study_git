# image : dog, cat, lion, suit

from tensorflow.keras.applications import VGG16
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

#1. DATA

img_dog = load_img('../data/image/vgg/dog2.jpg', target_size=(224,224))
img_cat = load_img('../data/image/vgg/cat2.jpg', target_size=(224,224))
img_lion = load_img('../data/image/vgg/lion2.jpg', target_size=(224,224))
img_suit = load_img('../data/image/vgg/suit2.jpg', target_size=(224,224))

# plt.imshow(img_suit)
# plt.show()

# print(img_suit) 
# <PIL.Image.Image image mode=RGB size=224x224 at 0x253E43885B0>
# array 형태로 바꿔야 한다.
arr_dog = img_to_array(img_dog)
arr_cat = img_to_array(img_cat)
arr_lion = img_to_array(img_lion)
arr_suit = img_to_array(img_suit)

print(arr_dog)
print(type(arr_dog))    # <class 'numpy.ndarray'>
print(arr_dog.shape)    # (224, 224, 3)

# 현재 RGB 상태임 --> VGG16에 넣기 위해서는 BGR로 바꿔줘야 한다.
from tensorflow.keras.applications.vgg16 import preprocess_input    # VGG16에 맞춰서 이미지를 알아서 변환해준다.
arr_dog = preprocess_input(arr_dog)
arr_cat = preprocess_input(arr_cat)
arr_lion = preprocess_input(arr_lion)
arr_suit = preprocess_input(arr_suit)

print(arr_dog)
print(arr_dog.shape)    # (224, 224, 3)

# 4개를 다 합쳐서 4차원으로 만들어준다.
arr_input = np.stack([arr_dog, arr_cat, arr_lion, arr_suit])
print(arr_input.shape)  # (4, 224, 224, 3)
# np.stack : 동일한 shape을 가진 배열들을 합친다.

#2. Modeling
model = VGG16()
results = model.predict(arr_input)

print(results)
print("results.shape : ", results.shape)

# [7.4410442e-09 3.2101872e-07 3.1724422e-11 ... 1.4191645e-10
#   1.9674411e-07 9.8373697e-08]                                    >> dog
#  [5.1106290e-06 1.3983141e-04 8.8099687e-06 ... 9.2090777e-06
#   7.1068917e-04 1.2132138e-03]                                    >> cat
#  [6.3565955e-07 2.3607799e-06 1.4053130e-06 ... 2.1121891e-06     
#   4.8312036e-06 8.3043335e-05]                                    >> lion
#  [1.5933378e-07 3.5637886e-08 2.7712071e-08 ... 6.3450476e-09
#   2.4350220e-07 1.0988249e-05]]                                   >> suit

# results.shape :  (4, 1000)                                        >> 1000 : imagenet에서 제공하는 카테고리 개수

#3. 이미지 결과 확인
from tensorflow.keras.applications.vgg16 import decode_predictions # 예측한 것을 해석하겠다. (결과를 보여주겠다.)

decode_results = decode_predictions(results)
# 예측한 결과 이름과 정확도를 보여줌
print("======================================")
print("reulsts[0] : ", decode_results[0]) 
print("======================================")
print("reulsts[1] : ", decode_results[1]) 
print("======================================")
print("reulsts[2] : ", decode_results[2]) 
print("======================================")
print("reulsts[3] : ", decode_results[3])
print("======================================")

# ======================================
# reulsts[0] :  [('n02112018', 'Pomeranian', 0.9909627), ('n02112137', 'chow', 0.0045381994), ('n02112350', 'keeshond', 0.001874596), ('n02123394', 'Persian_cat', 0.0010790719), ('n03803284', 'muzzle', 0.0004777925)]
# ======================================
# reulsts[1] :  [('n02123159', 'tiger_cat', 0.27818054), ('n02123045', 'tabby', 0.24903029), ('n02124075', 'Egyptian_cat', 0.23368542), ('n02883205', 'bow_tie', 0.040857907), ('n03124170', 'cowboy_hat', 0.014347721)]
# ======================================
# reulsts[2] :  [('n04548280', 'wall_clock', 0.25236833), ('n02708093', 'analog_clock', 0.078438014), ('n03532672', 'hook', 0.061394814), ('n03291819', 'envelope', 0.04905947), ('n04127249', 'safety_pin', 0.037014622)]
# ======================================
# reulsts[3] :  [('n04350905', 'suit', 0.95090723), ('n03594734', 'jean', 0.015000743), ('n04591157', 'Windsor_tie', 0.008821072), ('n02963159', 'cardigan', 0.0077682687), ('n02883205', 'bow_tie', 0.005021295)]
# ======================================