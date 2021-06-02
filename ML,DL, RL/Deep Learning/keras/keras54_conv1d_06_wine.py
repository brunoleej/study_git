# CNN으로 구성

# Moudule import 
import numpy as np
import tensorflow as tf
from sklearn.datasets import load_wine

wine = load_wine()
# print(wine.DESCR)
# print(wine.feature_names)

data = wine.data
target = wine.target
# print(data)
# print(target)
# print(data.shape) #(178,13)
# print(target.shape) #(178,)

# OneHotEncoding
from tensorflow.keras.utils import to_categorical
target = to_categorical(target)
print(target.shape) #(178,3)

# Preprocessing
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data,target, test_size = 0.3, random_state = 55)
print(x_train.shape,x_test.shape)   # (124, 13) (54, 13)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(124,13,1)
x_test = x_test.reshape(54,13,1)

# Modeling
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv1D(16,2,padding='SAME',activation='relu',input_shape=(13,1)),
    tf.keras.layers.Conv1D(32,2,activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(3,activation='softmax')
])

# Compile
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')

# Early Stopping
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor = 'acc', patience = 10, mode='max')

# Fit
model.fit(x_train, y_train, epochs=1000,callbacks=[early_stopping], validation_split=0.2, batch_size=8)

# Evaluate
loss, acc = model.evaluate(x_test, y_test, batch_size=8)
print('loss: ',loss)
print('acc: ',acc)

# Prediction
y_predict = model.predict(x_test[-5:-1])
print(y_predict)
print(np.argmax(y_predict,axis=-1))

# Dense Model
# loss:  1.0445780754089355
# accuracy:  0.7592592835426331

# CNN Model
# loss:  0.326945424079895
# acc:  0.9629629850387573

# Conv1D
# loss:  0.077336885035038
# acc:  0.9814814925193787