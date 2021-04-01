# XOR (최대 acc : 0.75)

import tensorflow as tf
import numpy as np

tf.set_random_seed(66)

x_data = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float32)  # (4, 2))
y_data = np.array([[0], [1], [1], [0]], dtype=np.float32) 

x = tf.placeholder(tf.float32, shape=[None,2])
y = tf.placeholder(tf.float32, shape=[None,1])

w = tf.Variable(tf.random_normal([2,1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# sigmoid
hypothesis = tf.sigmoid(tf.matmul(x, w) + b)   

# binary_crossentropy
cost = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis)) 

train = tf.train.GradientDescentOptimizer(learning_rate=0.05).minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted,y), dtype=tf.float32))

with tf.Session() as sess :
    sess.run(tf.global_variables_initializer())

    for step in range(5001) :
        cost_val, _ = sess.run([cost, train], feed_dict={x:x_data, y:y_data})

        if step % 200 == 0 :
            print(step, cost_val)

    h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={x:x_data, y:y_data})
    print("예측값 \n", h, "\n원래값 \n", c, "\nAccuracy ", a )

# 예측값 
#  [[0.5000038 ]
#  [0.5000006 ]
#  [0.5000006 ]
#  [0.49999738]]
# 원래값
#  [[1.]
#  [1.]
#  [1.]
#  [0.]]
# Accuracy  0.75