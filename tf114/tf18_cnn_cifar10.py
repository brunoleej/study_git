from tensorflow.keras.datasets import cifar10
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

tf.compat.v1.disable_eager_execution()
print(tf.executing_eagerly())   # False
print(tf.__version__)   # base 2.3.1


#1. DATA
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train/255.
x_test = x_test/255.
print(x_train.shape, x_test.shape)  # (50000, 32, 32, 3) (10000, 32, 32, 3)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train.shape, y_test.shape)  # (50000, 10) (10000, 10)

x = tf.compat.v1.placeholder(tf.float32, [None,32,32,3])
y = tf.compat.v1.placeholder(tf.float32, [None, 10])

#2. Modeling
# layer 1   # tf.keras.initializers.GlorotNormal()
# initializer=tf.compat.v1.initializers.he_normal()
w1 = tf.compat.v1.get_variable("w1", shape=[2,2,3,64], initializer=tf.compat.v1.initializers.he_normal())
L1 = tf.nn.conv2d(x, w1, strides=[1,1,1,1], padding='SAME')
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1,2,2,1], strides=[1,2,2,1],padding='SAME')
print(L1)   # Tensor("MaxPool:0", shape=(None, 16, 16, 128), dtype=float32)

# layer 2
w2 = tf.compat.v1.get_variable("w2", shape=[2,2,64,128], initializer=tf.compat.v1.initializers.he_normal())
L2 = tf.nn.conv2d(L1, w2, strides=[1,1,1,1], padding='SAME')
L2 = tf.nn.elu(L2)
# L2 = tf.nn.max_pool(L2, ksize=[1,2,2,1], strides=[1,2,2,1],padding='SAME')
print(L2)   # Tensor("MaxPool_1:0", shape=(None, 8, 8, 64), dtype=float32)

# layer 3
w3 = tf.compat.v1.get_variable("w3", shape=[2,2,128,64], initializer=tf.compat.v1.initializers.he_normal())
L3 = tf.nn.conv2d(L2, w3, strides=[1,1,1,1], padding='SAME')
L3 = tf.nn.elu(L3)
# L3 = tf.nn.max_pool(L3, ksize=[1,2,2,1], strides=[1,2,2,1],padding='SAME')
print(L3)   # Tensor("MaxPool_2:0", shape=(None, 8, 8, 32), dtype=float32))

# layer 4
w4 = tf.compat.v1.get_variable("w4", shape=[2,2,64,64])
L4 = tf.nn.conv2d(L3, w4, strides=[1,1,1,1], padding='SAME')
L4 = tf.nn.elu(L4)
# L4 = tf.nn.max_pool(L4, ksize=[1,2,2,1], strides=[1,2,2,1],padding='SAME')
print(L4)   # Tensor("MaxPool_3:0", shape=(None, 8, 8, 32), dtype=float32)

# Flatten
L_flat = tf.reshape(L4, [-1, 16*16*64])
print(L_flat)   # Tensor("Reshape:0", shape=(None, 128), dtype=float32)

# layer5
w5 = tf.compat.v1.get_variable("w5", shape=[16*16*64,64], initializer=tf.compat.v1.initializers.he_normal())
b5 = tf.Variable(tf.compat.v1.random_normal([64], stddev=0.1), name='b5')
L5 = tf.nn.elu(tf.matmul(L_flat, w5)+b5)
print(L5)   # Tensor("Elu:0", shape=(None, 64), dtype=float32)

# layer 6
w6 = tf.compat.v1.get_variable("w6", shape=[64,32], initializer=tf.compat.v1.initializers.he_normal())
b6 = tf.Variable(tf.compat.v1.random_normal([32], stddev=0.1), name='b6')
L6 = tf.nn.elu(tf.matmul(L5, w6)+b6)
print(L6)   #Tensor("Elu_1:0", shape=(None, 32), dtype=float32)

# layer 7 : output 
w7 = tf.compat.v1.get_variable("w7", shape=[32,10], initializer=tf.keras.initializers.GlorotNormal())
b7 = tf.Variable(tf.compat.v1.random_normal([10], stddev=0.1), name='b7')
hypothesis = tf.nn.softmax(tf.matmul(L6, w7)+b7)
print(hypothesis)   # Tensor("Softmax:0", shape=(None, 10), dtype=float32)

#3. Compile, Train
loss = tf.reduce_mean(-tf.reduce_mean(y*tf.math.log(hypothesis), axis=1)) # categorical_crossentropy
optimizer = tf.compat.v1.train.AdamOptimizer(1e-4).minimize(loss)

# Train
training_epochs = 28
batch_size = 100
total_batch = int(len(x_train)/batch_size)  # 60000 / 100 = 600

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

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

# epoch :  0025 loss = 0.021565290
# ACC :  0.6314
