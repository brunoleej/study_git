# VGG16으로 여자/남자 구별해보기


import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPooling2D, BatchNormalization, AveragePooling2D, Activation
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from tensorflow.keras.applications import VGG16, VGG19, EfficientNetB0
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from PIL import Image 


#1. DATA
# npy load
x_train = np.load('../data/image/gender/npy/keras67_train_x.npy')
x_valid = np.load('../data/image/gender/npy/keras67_valid_x.npy')
y_train = np.load('../data/image/gender/npy/keras67_train_y.npy')
y_valid = np.load('../data/image/gender/npy/keras67_valid_y.npy')

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, train_size=0.8, shuffle=True, random_state=42)

print(x_train.shape, x_valid.shape, x_test.shape)  # (1111, 56, 56, 3) (347, 56, 56, 3) (278, 56, 56, 3)
print(y_train.shape, y_valid.shape, y_test.shape)  # (1111,) (347,) (278,)

x_train = preprocess_input(x_train)
x_test = preprocess_input(x_test)
x_valid = preprocess_input(x_valid)

#2. Modeling
# apl = VGG16(weights='imagenet', include_top=False, input_shape=(56,56,3)) 
# apl = VGG19(weights='imagenet', include_top=False, input_shape=(56,56,3)) 
# apl = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(56,56,3)) 
# apl.trainable = False

# model = Sequential()
# model.add(apl)
# model.add(Flatten())
# model.add(Dense(32, activation='relu'))
# model.add(BatchNormalization())
# model.add(Dense(16, activation='relu'))
# model.add(BatchNormalization())
# model.add(Dense(1, activation='sigmoid'))

# model = load_model('../data/modelcheckpoint/k81_1_0.674.hdf5')  # VGG16
# model = load_model('../data/modelcheckpoint/k81_2_0.693.hdf5')  # VGG19
model = load_model('../data/modelcheckpoint/k81_3_0.693.hdf5')  # B0

#3. Compile, Train
es = EarlyStopping(monitor='val_loss', patience=20, mode='min')
lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=10, mode='min')
path = '../data/modelcheckpoint/k81_3_{val_loss:.3f}.hdf5'
cp = ModelCheckpoint(path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.01), metrics=['acc'])
# model.fit(x_train, y_train, epochs=1000, batch_size=4, validation_data=(x_valid, y_valid), callbacks=[es, lr, cp])

loss, acc = model.evaluate(x_test, y_test, batch_size=4)
print("loss : ", loss)
print("acc : ", acc)


datagen_2 = ImageDataGenerator(rescale=1./255)

# my image >> x_pred
im1 = Image.open('../data/image/gender/HHM.jpg')   # f
my1 = np.asarray(im1)
my1 = np.resize(my1, (56, 56, 3))
my1 = my1.reshape(1, 56, 56, 3)
my = datagen_2.flow(my1)

im2 = Image.open('../data/image/gender/HL2.jpg')   # f
my2 = np.asarray(im2)
my2 = np.resize(my2, (56, 56, 3))
my2 = my2.reshape(1, 56, 56, 3)
HL = datagen_2.flow(my2)

im3 = Image.open('../data/image/gender/LHL.jpg')   # f
my3 = np.asarray(im3)
my3 = np.resize(my3, (56, 56, 3))
my3 = my3.reshape(1, 56, 56, 3)
LHL = datagen_2.flow(my3)

im4 = Image.open('../data/image/gender/KDW.jpg')   # m
my4 = np.asarray(im4)
my4 = np.resize(my4, (56, 56, 3))
my4 = my4.reshape(1, 56, 56, 3)
KDW = datagen_2.flow(my4)

im5 = Image.open('../data/image/gender/HB.jpg')    # m
my5 = np.asarray(im5)
my5 = np.resize(my5, (56, 56, 3))
my5 = my5.reshape(1, 56, 56, 3)
HB = datagen_2.flow(my5)

######################

my_pred = model.predict(my)
my_pred = my_pred[0][0]
# print(my_pred)
print("당신은   ",np.round((1-my_pred)*100,2), '%의 확률로 여자입니다.')

HL_pred = model.predict(HL)
HL_pred = HL_pred[0][0]
# print(HL_pred)
print("이혜리는 ",np.round((1-HL_pred)*100,2), '%의 확률로 여자입니다.')

LHL_pred = model.predict(LHL)
LHL_pred = LHL_pred[0][0]
# print(LHL_pred)
print("이효리는 ",np.round((1-LHL_pred)*100,2), '%의 확률로 여자입니다.')

KDW_pred = model.predict(KDW)
KDW_pred = KDW_pred[0][0]
# print(KDW_pred)
print("강동원는 ",np.round(KDW_pred*100,2), '%의 확률로 남자입니다.')

HB_pred = model.predict(HB)
HB_pred = HB_pred[0][0]
# print(HB_pred)
print("현빈은   ",np.round(HB_pred*100,2), '%의 확률로 남자입니다.')

# VGG16
# loss :  0.6787249445915222
# acc :  0.6000000238418579
# 당신은    50.53 %의 확률로 여자입니다.
# 이혜리는  43.74 %의 확률로 여자입니다.
# 이효리는  38.59 %의 확률로 여자입니다.
# 강동원는  47.24 %의 확률로 남자입니다.
# 현빈은    51.58 %의 확률로 남자입니다.

# VGG19
# loss :  0.6911725401878357
# acc :  0.5395683646202087
# 당신은    47.66 %의 확률로 여자입니다.
# 이혜리는  47.86 %의 확률로 여자입니다.
# 이효리는  47.5 %의 확률로 여자입니다.
# 강동원는  52.77 %의 확률로 남자입니다.
# 현빈은    52.18 %의 확률로 남자입니다.

# B0
# loss :  0.6913778781890869
# acc :  0.5395683646202087
# 당신은    51.24 %의 확률로 여자입니다.
# 이혜리는  51.29 %의 확률로 여자입니다.
# 이효리는  51.27 %의 확률로 여자입니다.
# 강동원는  48.77 %의 확률로 남자입니다.
# 현빈은    48.75 %의 확률로 남자입니다.