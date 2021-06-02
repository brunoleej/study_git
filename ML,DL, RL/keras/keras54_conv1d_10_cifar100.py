import numpy as np
import tensorflow as tf
from keras.datasets import cifar100

(x_train,y_train),(x_test,y_test) = cifar100.load_data()
print(x_train.shape,x_test.shape)   # (50000, 32, 32, 3) (10000, 32, 32, 3)
x_train = x_train.reshape(50000,1024,3)
x_test = x_test.reshape(10000,1024,3)
print(np.min(x_train),np.max(x_test))   #  0 255
# print(y_train.shape,y_test.shape)   # (50000, 1) (10000, 1)

x_train,x_test = x_train / 255.0, x_test / 255.0
print(np.min(x_train),np.max(x_test))   # 0.0 1.0

# to_categorical
from keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Modeling
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv1D(16,2,padding='SAME',activation='relu',input_shape=(1024,3)),
    tf.keras.layers.Conv1D(32,2,activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(100,activation='softmax')
])
# Compile
model.compile(loss = 'categorical_crossentropy',metrics = ['acc'])

# fit
model.fit(x_train,y_train,batch_size = 32,shuffle=True,epochs = 10)

# Evaluate
loss,acc = model.evaluate(x_test,y_test,batch_size = 32)
print('loss : ',loss)
print('accuracy: ',acc)

# CNN batch_size = 64, epoch=5
# loss :  3.1846044063568115
# accuracy:  0.22349999845027924

# CNN batch_size = 32,epoch = 5
# loss :  3.255446195602417
# accuracy:  0.2142000049352646

# ConV1D
# loss :  4.624215126037598
# accuracy:  0.2354000061750422