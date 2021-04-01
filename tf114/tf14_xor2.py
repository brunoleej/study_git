# XOR (최대 acc : 0.75)
# 해결 : 레이어 수를 늘려준다. (acc : 1.0)

import tensorflow as tf
import numpy as np

tf.set_random_seed(66)

x_data = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float32)  # (4, 2))
y_data = np.array([[0], [1], [1], [0]], dtype=np.float32) 

x = tf.placeholder(tf.float32, shape=[None,2])
y = tf.placeholder(tf.float32, shape=[None,1])

# Node shape를 잘 맞춰줘야 한다. (input 2 개 -> node 10 개 -> node 7 개 -> output 1개)
# layer 1 : input
w1 = tf.Variable(tf.random_normal([2, 32]), name='weight1') # node 개수 10개 >> (None, 10)
b1 = tf.Variable(tf.random_normal([32]), name='bias1')      
layer1 = tf.sigmoid(tf.matmul(x, w1) + b1)
# model.add(Dense(10, input_dim=2, activation='sigmoid))

# layer 2
w2 = tf.Variable(tf.random_normal([32,16]), name='weight2')  # (None, 10) >> node 개수 7개 (10, 7)
b2 = tf.Variable(tf.random_normal([16]), name='bias2')
layer2 = tf.sigmoid(tf.matmul(layer1, w2) + b2)    # layer1의 값을 가져온다.
# model.add(Dense(7, activation='sigmoid))

# layer 3 : output
w3 = tf.Variable(tf.random_normal([16,1]), name='weight3')
b3 = tf.Variable(tf.random_normal([1]), name='bias3')
hypothesis = tf.sigmoid(tf.matmul(layer2, w3) + b3)   # (none,1)이 나와야 한다. # layer2의 값을 가녀와 최종 아웃풋을 계산한다.
# model.add(Dense(1, activation='sigmoid))

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

# Deep Layer 구성한 후
# 예측값 
#  [[0.0162153 ]
#  [0.9716693 ]
#  [0.98412293]
#  [0.02576704]]
# 원래값
#  [[0.]
#  [1.]
#  [1.]
#  [0.]]
# Accuracy  1.0