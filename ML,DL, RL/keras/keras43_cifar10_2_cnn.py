import numpy as np
import tensorflow as tf

EPOCHS = 100

# Network Architecture
class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel,self).__init__
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
        self.dense2 = tf.kears.layers.Dense(10,activation = 'softmax')

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

# Data
from keras.datasets import cifar10

(x_train,y_train),(x_test,y_test) = cifar10.load_data()
x_train,x_test = x_train / 255.0, x_test / 255.0

x_train = x_train.astype(np.float32)
x_test = x_train.astype(np.float32)

train_ds = tf.data.Dataset.from_tensor_slices((x_train,y_train)).shuffle(5000).batch(32).prefetch(512)
test_ds = tf.data.Dataset.from_tensor_slices((x_test,y_test)).batch(32).prefetch(512)

# # to_categorical
# from keras.utils import to_categorical
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)

# # Modeling
# from keras.layers import Dense,Conv2D,Dropout,Flatten,MaxPool2D,Input,Activation
# from keras.models import Model

# input1 = Input(shape = (32,32,3))
# net = Conv2D(32,3,padding='SAME')(input1)
# net = Activation('relu')(net)
# net = Conv2D(32,3,3,padding = 'SAME')(net)
# net = Activation('relu')(net)
# net = MaxPool2D(pool_size=(2,2))(net)
# net = Dropout(0.25)(net)

# net = Conv2D(64,3,padding='SAME')(net)
# net = Activation('relu')(net)
# net = Conv2D(64,3,3,padding = 'SAME')(net)
# net = Activation('relu')(net)
# net = MaxPool2D(pool_size=(2,2))(net)
# net = Dropout(0.25)(net)

# net = Flatten()(net)
# net = Dense(512)(net)
# net = Activation('relu')(net)
# net = Dropout(0.5)(net)
# net = Dense(10)(net)
# net = Activation('softmax')(net)

# model = Model(inputs = input1, outputs = net, name = 'sifar10_CNN_Model')

# Compile
model = MyModel()
model.compile(loss = 'sparse_categorical_crossentropy',metrics = ['acc'])

# fit
model.fit(train_ds,validation_data = test_ds,epochs = 5)

# # Evaluate
# loss,acc = model.evaluate(x_test,y_test,batch_size = 64)
# print('loss : ',loss)
# print('accuracy: ',acc)

# CNN
# loss :  1.1537483930587769
# accuracy:  0.585099995136261