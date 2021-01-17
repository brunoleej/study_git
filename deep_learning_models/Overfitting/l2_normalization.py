import numpy as np
import tensorflow as tf

EPOCHS = 100

class ConvLNReluBlock(tf.keras.Model):
    def __init__(self,filters,kernel_size):
        super(ConvLNReluBlock,self).__init__()
        self.conv = tf.keras.layers.Conv2D(filters,kernel_size,padding = 'same',use_bias=False)
        self.ln = tf.keras.layers.LayerNormalization()

    def call(self,x,training = False, mask = None):
        x = self.conv(x)
        x = self.ln(x)
        return tf.nn.relu(x)

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel,self).__init__()
        self.conv1_1 = ConvLNReluBlock(16,(3,3))
        self.conv1_2 = ConvLNReluBlock(16,(3,3))
        self.pool1 = tf.keras.layers.MaxPool2D((2,2))
        
        self.conv2_1 = ConvLNReluBlock(32,(3,3))
        self.conv2_2 = ConvLNReluBlock(32,(3,3))
        self.pool2 = tf.keras.layers.MaxPool2D((2,2))

        self.conv3_1 = ConvLNReluBlock(64,(3,3))
        self.conv3_2 = ConvLNReluBlock(64,(3,3))

        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(1024,activation = 'relu',kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.dense2 = tf.keras.layers.Dense(10,activation = 'softmax',kernel_regularizer=tf.keras.regularizers.l2(0.01))

    def call(self,x,training = False, mask = None):
        x  = self.conv1_1(x)
        x  = self.conv2_1(x)
        x  = self.poo1(x)

        x  = self.conv2_1(x)
        x  = self.conv2_2(x)
        x  = self.pool2(x)

        x  = self.conv3_1(x)
        x  = self.conv3_2(x)

        x  = self.flatten(x)
        x  = self.dense1(x)
        return self.dense2(x)

# Dataset
mnist = tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test) = mnist.load_data()

x_train,x_test = x_train / 255., x_test / 255.
x_train = x_train[...,tf.newaxis]
x_test = x_train[...,tf.newaxis]

train_ds = tf.data.Dataset.from_tensor_slices((x_train,y_train)).shuffle(10000).batch(32).prefetch(1024)
test_ds = tf.data.Dataset.from_tensor_slices((x_test,y_test)).batch(32).prefetch(1024)


# Model Evaluate
model = MyModel()
model.compile(loss = 'sparse_categorical_crossentropy',opimizer = 'adam',metrics = ['accuracy'])
model.fit(train_ds,validation_data = test_ds,epochs = EPOCHS)