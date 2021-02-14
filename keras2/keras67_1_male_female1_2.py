# male, female : Imagegenerator, fit_generator
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
    validation_split=0.2
)
test_datagen = ImageDataGenerator(rescale=1./255)

batch = 32

train = train_datagen.flow_from_directory(
    '../data/image/gender/',
    target_size=(56, 56),
    class_mode='binary',
    # batch_size=batch,
    batch_size=1389,
    subset="training"
)
# Found 1389 images belonging to 2 classes.

valid = train_datagen.flow_from_directory(
    '../data/image/gender/',
    target_size=(56, 56),
    class_mode='binary',
    # batch_size=batch,
    batch_size=347,
    subset="validation"
)
# Found 347 images belonging to 2 classes.

print(train[0][0].shape)    # (32, 56, 56, 3) => x_train
print(train[0][1].shape)    # (32,) => y_train

print(valid[0][0].shape)     # (32, 56, 56, 3) => x_valid
print(valid[0][1].shape)     # (32,) => x_valid

# data numpy save
np.save('../data/image/gender/npy/keras67_train_x.npy', arr = train[0][0]) 
np.save('../data/image/gender/npy/keras67_train_y.npy', arr = train[0][1]) 
np.save('../data/image/gender/npy/keras67_valid_x.npy', arr = valid[0][0]) 
np.save('../data/image/gender/npy/keras67_valid_y.npy', arr = valid[0][1]) 

'''
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

es = EarlyStopping(monitor='val_loss', patience=20, mode='min')
lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=10, mode='min')

# Compile
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
history = model.fit_generator(
    train, steps_per_epoch=1389//batch, epochs=100, validation_data=valid, validation_steps=4
)

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

print("acc : ", acc[-1])           
print("val_acc : ", val_acc[:-1])   

# acc :  0.5983787775039673
# val_acc :  [0.515625, 0.46875, 0.5078125, 0.515625, 0.515625, 0.484375, 0.453125, 0.46875, 0.46875, 0.6484375, 0.53125, 0.5625, 0.484375, 0.484375, 0.4921875, 0.5625, 0.5703125, 0.5703125, 0.5703125, 0.53125, 0.6796875, 
# 0.53125, 0.625, 0.5703125, 0.5, 0.5625, 0.5390625, 0.6015625, 0.609375, 0.609375, 0.5703125, 0.6171875, 0.5703125, 0.5859375, 0.4765625, 0.53125, 0.53125, 0.4921875, 0.609375, 0.671875, 0.5859375, 0.53125, 0.546875, 0.609375, 0.484375, 0.5703125, 0.5078125, 0.640625, 0.59375, 0.609375, 0.484375, 0.5234375, 0.6171875, 0.625, 0.5546875, 0.59375, 0.515625, 0.6953125, 0.6328125, 0.5703125, 0.578125, 0.5859375, 0.6328125, 0.515625, 0.578125, 0.6328125, 0.609375, 0.609375, 0.6015625, 0.5859375, 0.5703125, 0.6328125, 0.5546875, 0.515625, 0.609375, 0.6640625, 0.59375, 0.5625, 0.6171875, 0.4921875, 0.5625, 0.53125, 0.6015625, 0.6484375, 0.5625, 0.671875, 0.609375, 0.5859375, 0.515625, 0.5546875, 0.546875, 0.6015625, 0.609375, 0.5703125, 0.6328125, 0.5703125, 0.5703125, 0.5703125, 0.5703125]
'''