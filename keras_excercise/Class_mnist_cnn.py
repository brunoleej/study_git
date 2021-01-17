# VGG-16
import numpy as np
import tensorflow as tf

EPOCHS = 100

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel,self).__init__()
        self.conv1_1 = tf.keras.layers.Conv2D(16,(3,3),padding = 'same',activation = 'relu')
        self.conv1_2 = tf.keras.layers.Conv2D(16,(3,3),padding = 'same',activation = 'relu')
        self.pool1 = tf.keras.layers.MaxPool2D((2,2))

        self.conv2_1 = tf.keras.layers.Conv2D(32,(3,3),padding = 'same',activation = 'relu')
        self.conv2_2 = tf.keras.layers.Conv2D(32,(3,3),padding = 'same',activation = 'relu')
        self.pool2 = tf.keras.layers.MaxPool2D((2,2))
        
        self.conv3_1 = tf.keras.layers.Conv2D(64,(3,3),padding = 'same',activation = 'relu')
        self.conv3_2 = tf.keras.layers.Conv2D(64,(3,3),padding = 'same',activation = 'relu')
        
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(1024,activation = 'relu')
        self.dense2 = tf.keras.layers.Dense(10,activation = 'relu')

    def call(self,x,training = False,mask = None):
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.pool1(x)

        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.pool2(x)

        x = self.conv3_1(x)
        x = self.conv3_2(x)

        x = self.flatten(x)
        x = self.dense1(x)
        return self.dense2(x)

# Dataset
mnist = tf.keras.datasets.mnist

(x_train,y_train),(x_test,y_test) = mnist.load_data()
x_train, x_test = x_train / 255., x_test / 255.

x_train = x_train[...,tf.newaxis]
x_test = x_test[...,tf.newaxis]

x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float32)

train_ds = tf.data.Dataset.from_tensor_slices((x_train,y_train)).shuffle(10000).batch(32).prefetch(1024)
test_ds = tf.data.Dataset.from_tensor_slices((x_test,y_test)).batch(32).prefetch(1024)

# Model evaluate
model = MyModel()
model.compile(loss = 'sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(train_ds,validation_data=test_ds,epochs=EPOCHS)