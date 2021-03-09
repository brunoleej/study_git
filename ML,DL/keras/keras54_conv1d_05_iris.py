# CNN으로 구성

# Moudule import
import numpy as np
import tensorflow as tf
from sklearn.datasets import load_iris

# Data
iris = load_iris()
data = iris.data
target = iris.target
print(data.shape) # (150,4)
print(target.shape) # (150,)

# OneHotEncoder
from tensorflow.keras.utils import to_categorical
target = to_categorical(target)

# Preprocessing
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data, target, train_size = 0.8, random_state = 66)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],1)


# Modeling
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv1D(16,2,padding='SAME',activation='relu',input_shape=(4,1)),
    tf.keras.layers.Conv1D(32,2,activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(3,activation='softmax')
])

# Compile
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

# EarlyStopping
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor = 'acc', patience=10, mode = 'max' )

# Fit
model.fit(x_train, y_train,epochs= 1000, callbacks=[early_stopping], validation_split=0.2, batch_size=8)

# Evaluate
loss,acc = model.evaluate(x_test, y_test, batch_size=8)
print('loss: ',loss)
print('acc: ',acc)

# Prediction
y_predict = model.predict(x_test[-5:-1])
print(y_predict)
print(y_test[-5:-1])
print(np.argmax(y_predict,axis=-1))

# Dense Model
# loss:  0.26552248001098633
# accuracy:  0.9111111164093018

# CNN model
# loss:  0.05330286920070648
# acc:  0.9777777791023254

# Conv1D
# loss:  0.12024890631437302
# acc:  1.0