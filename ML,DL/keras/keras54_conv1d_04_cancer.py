# CNN으로 구성
import numpy as np
import tensorflow as tf
from sklearn.datasets import load_breast_cancer

# Data
cancer = load_breast_cancer()
data = cancer.data
target = cancer.target
# print(cancer.DESCR)
# print(cancer.feature_names)

print(data.shape) # (569, 30)
print(target.shape)  # (569,) 

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
    tf.keras.layers.Conv1D(16,2,padding='SAME',activation='relu',input_shape=(30,1)),
    tf.keras.layers.Conv1D(32,2,activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(1,activation='sigmoid')
])

# Compile
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

# Early Stopping
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor = 'acc', patience=10, mode = 'max' )

# Fit
model.fit(x_train,y_train, epochs=3000, callbacks=[early_stopping], validation_split=0.2, batch_size=8)

# Evaluate
loss, acc = model.evaluate(x_test,y_test, batch_size=8)
print('loss: ',loss)
print('acc: ',acc)

y_predict = model.predict(x_test[-5:-1])
print(y_predict)
print(y_test[-5:-1])
print(np.argmax(y_predict,axis=0))

# Dense Model
# loss:  0.2283373773097992
# accuracy:  0.9181286692619324

# CNN Model
# loss:  0.22723810374736786
# accuracy:  0.9707602262496948

# Conv1D
# loss:  0.06368209421634674
# acc:  0.9824561476707458