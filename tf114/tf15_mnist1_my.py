# mnist

import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score

tf.set_random_seed(166)

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)
# (60000, 28, 28) (10000, 28, 28)
# (60000,) (10000,)

# preprocess

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1]*x_train.shape[2])/255.
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1]*x_test.shape[2])/255.
print(x_train.shape, x_test.shape)  # (60000, 784) (10000, 784)

y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)

encoder = OneHotEncoder()
encoder.fit(y_train)
encoder.fit(y_test)
y_train = encoder.transform(y_train).toarray()
# y_test = encoder.transform(y_test).toarray()
print(y_train.shape, y_test.shape)  # (60000, 10) (10000, 10)

####
x = tf.placeholder('float', [None,784])
y = tf.placeholder('float', [None,10])

n1 = 256
n2 = 64
n3 = 16

# layer1
w1 = tf.Variable(tf.random.normal([784,n1],stddev= 0.1,name='weight1'))
b1 = tf.Variable(tf.random.normal([1,n1],stddev= 0.1,name='bias1'))
layer1 = tf.nn.relu(tf.matmul(x,w1)+b1) # activation='relu'

# layer2
w2 = tf.Variable(tf.random.normal([n1,n2],stddev= 0.1,name='weight2'))
b2 = tf.Variable(tf.random.normal([1,n2],stddev= 0.1,name='bias2'))
layer2 = tf.nn.relu(tf.matmul(layer1,w2)+b2)

# layer3
w3 = tf.Variable(tf.random.normal([n2,n3],stddev= 0.1,name='weight3'))
b3 = tf.Variable(tf.random.normal([1,n3],stddev= 0.1,name='bias3'))
layer3 = tf.nn.relu(tf.matmul(layer2,w3)+b3)

# layer4
w4 = tf.Variable(tf.random.normal([n3,10],stddev= 0.1,name='weight4'))
b4 = tf.Variable(tf.random.normal([1,10],stddev= 0.1,name='bias4'))
hypothesis = tf.nn.softmax(tf.matmul(layer3,w4)+b4)

loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1))    # categorical_crossentropy

# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.8).minimize(loss)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)

# prediction = tf.equal(tf.argmax(hypothesis, axis=1), tf.argmax(y, axis=1))
# accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))

with tf.Session() as sess :
    sess.run(tf.global_variables_initializer())
    for step in range(201) :
        _, cost_val = sess.run([optimizer, loss], feed_dict={x:x_train, y:y_train})
        if step % 20 == 0 :
            y_pred = sess.run(hypothesis, feed_dict = {x:x_test})
            y_pred = np.argmax(y_pred, axis=1)
            print(step, "/ loss : ", cost_val, '/ acc : ', accuracy_score(y_test, y_pred))

    y_predict = sess.run(hypothesis, feed_dict={x:x_test})
    predicted = sess.run(tf.argmax(y_predict,1))
    # test_data = sess.run(tf.argmax(y_test, 1))
    print("accuracy_score : ", accuracy_score(y_test, predicted))

# 200 / loss :  0.08619085 / acc :  0.9643
# accuracy_score :  0.9643