# keras45_mnist2_cnn.py copy()
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

EPOCHS = 5
batch_size = 64

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel,self).__init__()
        self.dense1 = tf.keras.layers.Dense(1024,activation = 'relu')
        self.dense2 = tf.keras.layers.Dense(512,activation = 'relu')
        self.drop1 = tf.keras.layers.Dropout(0.25)
        self.dense3 = tf.keras.layers.Dense(50,activation = 'relu')
        self.dense4 = tf.keras.layers.Dense(10,activation = 'softmax')
    def call(self,x,training = False,mask = None):
        
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.drop1(x)
        x = self.dense3(x)
        return self.dense4(x)

# Data
mnist = tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test) = mnist.load_data()

x_train = tf.cast(x_train,dtype='float32')
x_test = tf.cast(x_test,dtype='float32')

# x_train = x_train[...,tf.newaxis]
# x_test = x_test[...,tf.newaxis]


x_train = x_train / 255.
x_test = x_test/255.

train_ds = tf.data.Dataset.from_tensor_slices((x_train,y_train)).shuffle(5000).batch(32).prefetch(512)
test_ds = tf.data.Dataset.from_tensor_slices((x_test,y_test)).batch(32).prefetch(512)

# Model
model = MyModel()

# EarlyStopping
from keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau
modelpath = '../modelCheckpoint/k57_mnist_{epoch:02d}-{val_loss:.4f}.hdf5'
early_stopping = EarlyStopping(monitor='val_loss',patience = 6,mode = 'auto')
check_point = ModelCheckpoint(filepath=modelpath,monitor='val_loss',save_best_only=True,mode='auto')
reduce_lr = ReduceLROnPlateau(monitor = 'val_loss',patience=3,factor=0.5,verbose=1)

# Compile
model.compile(loss = 'sparse_categorical_crossentropy',metrics = ['acc'])

# fit
hist = model.fit(train_ds,validation_data = test_ds,epochs = EPOCHS,batch_size = batch_size,callbacks=[early_stopping,check_point,reduce_lr])

# Evaluate
loss = model.evaluate(train_ds,test_ds,batch_size = batch_size)
print('loss : ',loss[0])
print('accuracy: ',loss[1])

# visualization
import matplotlib.pyplot as plt
import matplotlib 
matplotlib.rcParams['axes.unicode_minus'] = False 
matplotlib.rcParams['font.family'] = "AppleGothic"

plt.figure(figsize = (10,6))
plt.subplot(211)    # 2 row 1 column
plt.plot(hist.history['loss'],marker = '.',c='red',label = 'loss')
plt.plot(hist.history['val_loss'],marker = '.',c='blue',label = 'val_loss')
plt.grid()

plt.title('Cost')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')

plt.subplot(212)    # 2 row 2 column
plt.plot(hist.history['acc'],marker = '.',c='red')
plt.plot(hist.history['val_acc'],marker = '.',c='blue')
plt.grid()

plt.title('Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['accuracy','val_accuracy'])
plt.show()

# loss :  0.03266778588294983
# accuracy:  0.989799976348877