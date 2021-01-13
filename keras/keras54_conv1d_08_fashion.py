import numpy as np
import tensorflow as tf
from keras.datasets import fashion_mnist

# preprocessing
(train_images,train_labels),(test_images,test_labels) = fashion_mnist.load_data()
print(train_images.shape,test_images.shape) # (60000, 28, 28) (10000, 28, 28)
print(train_labels.shape,test_labels.shape) # (60000,) (10000,)
print(np.min(train_images),np.max(test_images)) # 0 255

train_images,test_images = train_images / 255.0, test_images / 255.0
print(np.min(train_images),np.max(test_images)) # 0.0 1.0
train_images = train_images.reshape(60000,784,1)
test_images = test_images.reshape(10000,784,1)
print(train_images.shape,test_images.shape) # (60000, 784, 1) (10000, 784, 1)

# to_categorical
from tensorflow.keras.utils import to_categorical
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Modeling
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv1D(16,2,padding='SAME',activation='relu',input_shape=(784,1)),
    tf.keras.layers.Conv1D(32,2,activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(10,activation='softmax')
])

# Compile
model.compile(loss = 'categorical_crossentropy',optimizer='adam',metrics = ['acc'])

# fit
model.fit(train_images,train_labels,epochs=5,batch_size=64)

# evaluate
loss,acc = model.evaluate(test_images,test_labels,batch_size = 64)
print('loss: ', loss)
print('acc: ', acc)

# CNN
# loss:  0.20688886940479279
# acc:  0.9332000112533569

# Conv1D
# loss:  0.29125428199768066
# acc:  0.9064000248908997