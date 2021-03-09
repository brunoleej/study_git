import numpy as np
import tensorflow as tf
# import matplotlib.pyplot as plt
from keras.datasets import mnist

(x_train,y_train),(x_test,y_test) = mnist.load_data()
print(x_train.shape,y_train.shape)  # (60000, 28, 28) (60000,)
print(x_test.shape,y_test.shape)    # (10000, 28, 28) (10000,)    
# print(np.min(x_train),np.max(x_test))   # 0 255

# Preprocessing
x_train = x_train.reshape(60000,784,1)
x_test = x_test.reshape(10000,784,1)

x_train = x_train / 255.
x_test = x_test/255.
# OneHotEncoding(to_categorical)
# from keras.utils import to_categorical

# Model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv1D(16,2,padding='SAME',activation='relu',input_shape=(784,1)),
    tf.keras.layers.Conv1D(32,2,activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(10,activation='softmax')
])

# Compile
model.compile(loss = 'sparse_categorical_crossentropy',metrics = ['acc'])

# fit
model.fit(x_train,y_train,batch_size = 64,shuffle=True,epochs = 5)

# Evaluate
loss,acc = model.evaluate(x_test,y_test,batch_size = 64)
print('loss : ',loss)
print('accuracy: ',acc)

# Conv1D
# loss :  0.07511574029922485
# accuracy:  0.9790999889373779