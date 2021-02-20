# 예측값 메일로 제출
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPool2D, BatchNormalization, AveragePooling2D, Activation
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.optimizers import Adam
import PIL.Image as pilimg
from PIL import Image
from sklearn.metrics import accuracy_score

# Data
# npy load
x_train = np.load('../data/image/gender/npy/keras67_train_x.npy')
x_valid = np.load('../data/image/gender/npy/keras67_valid_x.npy')
y_train = np.load('../data/image/gender/npy/keras67_train_y.npy')
y_valid = np.load('../data/image/gender/npy/keras67_valid_y.npy')

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, train_size=0.95, shuffle=True, random_state=42)

print(x_train.shape, x_valid.shape, x_test.shape)  # (1319, 56, 56, 3) (347, 56, 56, 3) (70, 56, 56, 3)
print(y_train.shape, y_valid.shape, y_test.shape)  # (1319,) (347,) (70,)

# Modeling
# model = load_model('../data/modelcheckpoint/k67_56_0.693.h5')   # 
# model = load_model('../data/modelcheckpoint/k67_56_2_0.691.h5')   # 
# model = load_model('../data/modelcheckpoint/k67_56_3_0.688.h5')   # 
# model = load_model('../data/modelcheckpoint/k67_56_6_0.690.hdf5')   # keep
# model = load_model('../data/modelcheckpoint/k67_0.682_17.h5')   # 
model = load_model('../data/modelcheckpoint/k67_56_7_0.674.hdf5')   # keep

# es = EarlyStopping(monitor='val_loss', patience=50, mode='min')
# lr = ReduceLROnPlateau(monitor='val_loss', factor=0.4, patience=30, mode='min')

# Compile
# model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.01), metrics=['acc'])

# Fitting
# model.fit(x_train, y_train, epochs=1000, batch_size=8, validation_data=(x_valid, y_valid), callbacks=[es, lr])

# Evaluate
loss, acc = model.evaluate(x_test, y_test, batch_size=16)
print("loss : ", loss)
print("acc : ", acc)

####################[My Image]####################

datagen_2 = ImageDataGenerator(rescale=1./255)

# my image >> x_pred
im1 = Image.open('../data/image/gender/HHM.jpg')   # f
my1 = np.asarray(im1)
my1 = np.resize(my1, (56, 56, 3))
my1 = my1.reshape(1, 56, 56, 3)
my = datagen_2.flow(my1)
######################

my_pred = model.predict(my)
my_pred = my_pred[0][0]
# print(my_pred)
print("당신은 ",np.round((1-my_pred)*100,2), '%의 확률로 여자입니다.')

# 당신은 52.51 %의 확률로 남자입니다.

