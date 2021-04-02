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

x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float', [None, 10])

w = tf.Variable(tf.random_normal([784, 10]), name='weight')
b = tf.Variable(tf.random_normal([10]), name='bias')

#2. Modeling
hypothesis = tf.nn.softmax((tf.matmul(x, w)+b))

#3. Compile, Train

loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1))    # categorical_crossentropy
optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)

with tf.Session() as sess :
    sess.run(tf.global_variables_initializer())

    for step in range(2001) :
        _, cost_val = sess.run([optimizer, loss], feed_dict={x:x_train, y:y_train})
        if step % 20 == 0 :
            print(step, " / loss : ", cost_val)
    # predict
    y_pred = sess.run(tf.argmax(hypothesis, axis=1), feed_dict={x:x_test})
    y_test_arg = sess.run(tf.argmax(y_test, 1))
    print("acc score : ", accuracy_score(y_test_arg, y_pred))
# 2000  / loss :  0.5915186
# acc score :  0.8629

