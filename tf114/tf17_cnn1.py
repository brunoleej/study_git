# cnn mnist

import numpy as np
import tensorflow as tf

tf.set_random_seed(66)

#1. DATA
from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, x_test.shape)  # (60000, 28, 28) (10000, 28, 28)
print(y_train.shape, y_test.shape)  # (60000,) (10000,)

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train.shape, y_test.shape)  # (60000, 10) (10000, 10)

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255.
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255.

learning_rate = 0.001
training_epochs = 15
batch_size = 100
total_batch = int(len(x_train)/batch_size)  # 60000 / 100 = 600

x = tf.placeholder(tf.float32, [None, 28, 28, 1])
y = tf.placeholder(tf.float32, [None, 10])

#2. Modeling
n1 = 128
n2 = 64
n3 = 32
n4 = 32

# layer 1
w1 = tf.get_variable("w1", shape=[3, 3, 1, 32]) # 4차원 맞춘다. : 3,3 kernel_size / 1 channel / 32 filter
L1 = tf.nn.conv2d(x, w1, strides=[1,1,1,1], padding='SAME') # strides 4차원을 맞춘다. : 1 [1,1,1,1] / 2 [1,2,2,1] / ... shape을 맞추기 위해서 앞 뒤로 1을 넣는다.
print(L1)   # Tensor("Conv2D:0", shape=(?, 28, 28, 32), dtype=float32)
# Conv2D summary 
# Conv2D (filter, (kernel_size), input_shape)
# parameter = ( input_dim * kernel_size + bias ) * filter
# Conv2D (32, (3,3), input_shape=(28,28,1)) 
# output : (None, 28, 28, 32)
L1 = tf.nn.selu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME') 
# output : (None, 14, 14, 32)
print(L1)   # Tensor("MaxPool:0", shape=(?, 14, 14, 32), dtype=float32)

# layer 2
w2 = tf.get_variable("w2", shape=[3, 3, 32, 64]) # 4차원 맞춘다. : 3,3 kernel_size / 32 위의 output이 현재 레이어의 input  / 32 filter
L2 = tf.nn.conv2d(L1, w2, strides=[1,1,1,1], padding='SAME')
L2 = tf.nn.elu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME') 
# output : (None, 7, 7, 64)
print(L2)   # Tensor("MaxPool_1:0", shape=(?, 7, 7, 64), dtype=float32)

# layer 3
w3 = tf.get_variable("w3", shape=[3, 3, 64, 128]) 
L3 = tf.nn.conv2d(L2, w3, strides=[1,1,1,1], padding='SAME')
L3 = tf.nn.relu(L3)
L3 = tf.nn.max_pool(L3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME') 
# output : (None, 4, 4, 128)
print(L3)  # Tensor("MaxPool_2:0", shape=(?, 4, 4, 128), dtype=float32)

# layer 4
w4 = tf.get_variable("w4", shape=[3, 3, 128, 64]) 
L4 = tf.nn.conv2d(L3, w4, strides=[1,1,1,1], padding='SAME')
L4 = tf.nn.selu(L4)
L4 = tf.nn.max_pool(L4, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME') 
# output : (None, 2, 2, 64)
print(L4)   # Tensor("MaxPool_3:0", shape=(?, 2, 2, 64), dtype=float32)

# Flatten
L_flat = tf.reshape(L4, [-1,2*2*64])
print(L_flat)   # Tensor("Reshape:0", shape=(?, 256), dtype=float32)

# layer 5
w5 = tf.get_variable("w5", shape=[2*2*64,64], 
                    initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([64]), name='b5')
L5 = tf.nn.selu(tf.matmul(L_flat, w5)+b5)
L5 = tf.nn.dropout(L5, keep_prob=0.2)
print(L5)   # Tensor("dropout/mul_1:0", shape=(?, 64), dtype=float32)

# Layer 6
w6 = tf.get_variable("w6", shape=[64,32], 
                    initializer=tf.contrib.layers.xavier_initializer())
b6 = tf.Variable(tf.random_normal([32]), name='b6')
L6 = tf.nn.selu(tf.matmul(L5, w6)+b6)
L6 = tf.nn.dropout(L6, keep_prob=0.2)
print(L6)   # Tensor("dropout_1/mul_1:0", shape=(?, 32), dtype=float32)

# Layer7 : output layer 
w7 = tf.get_variable("w7", shape=[32,10], 
                    initializer=tf.contrib.layers.xavier_initializer())
b7 = tf.Variable(tf.random_normal([10]), name='b7')
hypothesis = tf.nn.softmax(tf.matmul(L6, w7)+b7)
print(hypothesis)   # Tensor("Softmax:0", shape=(?, 10), dtype=float32)

#3. Compile, Train
loss = tf.reduce_mean(-tf.reduce_mean(y*tf.log(hypothesis), axis=1)) # categorical_crossentropy
optimizer = tf.train.AdamOptimizer(0.01).minimize(loss)

# Train
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epoch in range(training_epochs):
    avg_loss= 0

    for i in range(total_batch):  # 600번 돈다
        start = i * batch_size
        end = start + batch_size

        batch_x, batch_y = x_train[start:end], y_train[start:end]
        feed_dict = {x:batch_x, y:batch_y}
        l, _ = sess.run([loss,optimizer], feed_dict=feed_dict)
        avg_loss += l/total_batch
    print('epoch : ', '%04d' %(epoch+1),
          'loss = {:.9f}'.format(avg_loss))


print("== Done ==")

#4. Evaluate, Predict
prediction = tf.equal(tf.argmax(hypothesis,1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))

print("ACC : ", sess.run(accuracy, feed_dict={x:x_test, y:y_test}))
