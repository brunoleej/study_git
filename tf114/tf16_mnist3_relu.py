# dropout 적용, 다층 레이어 적용

import tensorflow as tf
import numpy as np

from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score

#1. DATA
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
print(x_test.shape, y_train.shape) # (60000, 28, 28) (60000,)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000, 28*28).astype('float32')/255
x_test = x_test.reshape(10000, 28*28).astype('float32')/255
print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000, 10)

#2. Modeling

x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float', [None, 10])

w = tf.Variable(tf.random_normal([784, 100], stddev=0.1), name='weight1')   # # stddev=0.1 랜덤한 숫자의 분포를 10%로. 디폴트는 1
b = tf.Variable(tf.random_normal([100],stddev= 0.1) ,name='bias')
# layer1 = tf.nn.softmax((tf.matmul(x, w)+b))   # activaion=softmax
# layer1 = tf.nn.relu((tf.matmul(x, w)+b))  # activation=relu
# layer1 = tf.nn.selu((tf.matmul(x, w)+b))  # activation=selu   
layer1 = tf.nn.elu((tf.matmul(x, w)+b)) # activation=elu
layer1 = tf.nn.dropout(layer1, keep_prob=0.3)   # keep_prob=0.3 : 30% drop out

w2 = tf.Variable(tf.random_normal([100,64],stddev= 0.1), name='weight2')
b2 = tf.Variable(tf.random_normal([64],stddev= 0.1), name='bias2')
layer2 = tf.nn.relu(tf.matmul(layer1, w2) + b2)

w3 = tf.Variable(tf.random_normal([64,10],stddev= 0.1), name='weight3')
b3 = tf.Variable(tf.random_normal([10],stddev= 0.1), name= 'bias3')
hypothesis = tf.nn.softmax(tf.matmul(layer2, w3)+b3)    # activation=softmax

#3. Compile, Train
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1))    # categorical_crossentropy
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

with tf.Session() as sess :
    sess.run(tf.global_variables_initializer())

    for step in range(201) :
        _, cost_val = sess.run([optimizer, loss], feed_dict={x:x_train, y:y_train})
        if step % 20 == 0 :
            print(step, "/ loss : ", cost_val)

    # predict
    y_pred = sess.run(tf.argmax(hypothesis, axis=1), feed_dict={x:x_test})
    y_test_arg = sess.run(tf.argmax(y_test, 1))
    print("acc score : ", accuracy_score(y_test_arg, y_pred))

# 200 / loss :  0.33884966
# acc score :  0.8995