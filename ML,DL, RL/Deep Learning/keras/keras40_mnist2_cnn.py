import numpy as np
import tensorflow as tf

EPOCHS = 5

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
        self.dense2 = tf.keras.layers.Dense(10,activation = 'softmax')

    def call(self,x,training = False, mask = None):
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

# Data
mnist = tf.keras.datasets.mnist

(x_train,y_train),(x_test,y_test) = mnist.load_data()

# preprocessing
x_train,x_test = x_train / 255., x_test / 255.
x_train = x_train[...,tf.newaxis]
x_test = x_test[...,tf.newaxis]

x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float32)

train_ds = tf.data.Dataset.from_tensor_slices((x_train,y_train)).shuffle(10000).batch(32).prefetch(1024)
test_ds = tf.data.Dataset.from_tensor_slices((x_test,y_test)).batch(32).prefetch(1024)

# Model
model = MyModel()
model.compile(loss = 'sparse_categorical_crossentropy',optimizer='adam',metrics = ['accuracy'])
model.fit(train_ds,validation_data=test_ds,epochs = EPOCHS)

# 실습!! 완성하시오.
# 지표는 acc /// 0.985 이상
# loss: 0.2328 - accuracy: 0.9264 - val_loss: 0.0482 - val_accuracy: 0.9844
# loss: 0.0466 - accuracy: 0.9856 - val_loss: 0.0374 - val_accuracy: 0.9889
# loss: 0.0335 - accuracy: 0.9899 - val_loss: 0.0352 - val_accuracy: 0.9892
# loss: 0.0258 - accuracy: 0.9914 - val_loss: 0.0274 - val_accuracy: 0.9916
# oss: 0.0205 - accuracy: 0.9934 - val_loss: 0.0244 - val_accuracy: 0.9920