import numpy as np

data = np.load('./data/iris_data.npy')
target = np.load('./data/iris_target.npy')

# Preprocessing
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(data,target,train_size = 0.7,shuffle = True)

from tensorflow.keras.utils import to_categorical
# from keras.utils.np_utils import to_categorical

# y = to_categorical(y)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(y_train)
print(x_train.shape)  # (105,4)
print(y_train.shape)  # (105,3)

# Model
from tensorflow.keras.layers import Dense,Input,Activation
from tensorflow.keras.models import Model

input1 = Input(shape=(4,))
dense1 = Dense(128)(input1)
dense1 = Dense(128)(dense1)
dense1 = Activation('relu')(dense1)
dense1 = Dense(64)(dense1)
dense1 = Dense(64)(dense1)
dense1 = Dense(3)(dense1)
dense1 = Activation('softmax')(dense1)
model = Model(inputs = input1,outputs = dense1)

# Model Compile
model.compile(loss = 'categorical_crossentropy',optimizer = 'adam',metrics = ['acc'])

# EarlyStopping
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor = 'acc',patience=20,mode = 'auto')

model.fit(x_train,y_train,epochs = 100, validation_split =0.2,callbacks = [early_stopping])

# Evaluate
loss,acc = model.evaluate(x_test,y_test)
print('loss: ',loss)
print('accuracy: ',acc)

# loss:  0.06167419254779816
# accuracy:  0.9777777791023254