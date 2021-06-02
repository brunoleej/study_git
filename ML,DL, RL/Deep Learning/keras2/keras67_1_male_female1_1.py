# male, female => Imagegenerator, fit_generator 
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, MaxPool2D, Flatten, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Data
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rotation_range=2,
    zoom_range=1.0,
    shear_range=0.5,
    fill_mode='nearest',
)
test_datagen = ImageDataGenerator(rescale=1./255)

train = train_datagen.flow_from_directory(
    '../data/image/gender/',
    target_size=(56, 56),
    # batch_size=10000,
    batch_size=217,
    class_mode='binary'
)
# Found 1736 images belonging to 2 classes.

test = test_datagen.flow_from_directory(
    '../data/image/gender/',
    target_size=(56, 56),
    # batch_size=10000,
    batch_size=217,
    class_mode='binary'
)
# Found 1736 images belonging to 2 classes.

print(train[0][0].shape)    # (217, 56, 56, 3) => x_train
print(train[0][1].shape)    # (217,) => y_train

print(test[0][0].shape)     # (217, 56, 56, 3) => x_test
print(test[0][1].shape)     # (217,) => y_test

# data numpy save
# np.save('../data/image/gender/npy/keras67_train_x.npy', arr = train[0][0]) 
# np.save('../data/image/gender/npy/keras67_train_y.npy', arr = train[0][1]) 
# np.save('../data/image/gender/npy/keras67_test_x.npy', arr = test[0][0]) 
# np.save('../data/image/gender/npy/keras67_test_y.npy', arr = test[0][1]) 

# Modeling
model = Sequential()
model.add(Conv2D(32, (2,2), padding='same', activation='relu', input_shape=(56, 56, 3)))
model.add(BatchNormalization())
model.add(Conv2D(32, (2,2), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Conv2D(64, (2,2), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(64, (2,2), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Conv2D(128, (2,2), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(128, (2,2), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile
es = EarlyStopping(monitor='val_loss', patience=20, mode='min')
lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=10, mode='min')

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit_generator(
    train, steps_per_epoch=8, epochs=100, validation_data=test, validation_steps=4
)

# Evaluate
loss, acc = model.evaluate(test)
print("loss : ", loss)
print("acc : ", acc)

# loss :  0.5482652187347412
# acc :  0.7050691246986389
