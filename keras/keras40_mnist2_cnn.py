import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from keras.datasets import mnist

(x_train,y_train),(x_test,y_test) = mnist.load_data()
print(x_train.shape,y_train.shape)  # (60000, 28, 28) (60000,)
print(x_test.shape,y_test.shape)    # (10000, 28, 28) (10000,)    
print(np.min(x_train),np.max(x_test))   # 0 255

print(x_train[0])
print('y_train[0] : ',y_train[0])
print(x_train[0].shape) # (28, 28)

# plt.imshow(x_train[0],'jet')
# plt.colorbar()
# plt.show()

# Preprocessing
x_train = tf.cast(x_train,dtype='float32')
x_train = x_train[...,tf.newaxis]
x_train = x_train / 255.
print(x_train[0].shape) # (28, 28, 1)

x_test = x_test[...,tf.newaxis]
x_test = x_test/255.
# (x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2], 1))
test_image = x_test[0,:,:,0]    # (28, 28)
print(test_image.shape)

# OneHotEncoding(to_categorical)
# from keras.utils import to_categorical

# Model Declaration
from keras.layers import Conv2D,MaxPool2D,Dense,Flatten,Dropout,Input,Activation
from keras.models import Model

# Fully Connected
input1 = Input(shape=(x_train[0].shape))
net = Conv2D(32,3,padding='SAME')(input1)
net = Activation('relu')(net)
net = Conv2D(32,3,3,padding = 'SAME')(net)
net = Activation('relu')(net)
net = MaxPool2D(pool_size=(2,2))(net)
net = Dropout(0.25)(net)

net = Conv2D(64,3,padding='SAME')(net)
net = Activation('relu')(net)
net = Conv2D(64,3,3,padding = 'SAME')(net)
net = Activation('relu')(net)
net = MaxPool2D(pool_size=(2,2))(net)
net = Dropout(0.25)(net)

net = Flatten()(net)
net = Dense(512)(net)
net = Activation('relu')(net)
net = Dropout(0.5)(net)
net = Dense(10)(net)
net = Activation('softmax')(net)

model = Model(inputs = input1, outputs = net, name = 'CNN_Model')

# Compile
model.compile(loss = 'sparse_categorical_crossentropy',metrics = ['acc'])

# fit
model.fit(x_train,y_train,batch_size = 64,shuffle=True,epochs = 5)

# Evaluate
loss,acc = model.evaluate(x_test,y_test,batch_size = 64)
print('loss : ',loss)
print('accuracy: ',acc)

# 실습!! 완성하시오.
# 지표는 acc /// 0.985 이상
# loss :  0.04032869264483452
# accuracy : 0.987500011920929

# 응용 y_test 10개와 y_pred 10개를 출력하시오
# pred = model.predict(test_image)
# print('y_pred',pred)

# y_test[:10] = (?,?,?,?,?,?,?,?,?,?)
# y_pred[:10] = (?,?,?,?,?,?,?,?,?,?)
