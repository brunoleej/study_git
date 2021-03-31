# Softmax

import tensorflow as tf
import numpy as np

tf.set_random_seed(66)

x_data = [[1,2,1,1],
          [2,1,3,2],
          [3,1,3,4],
          [4,1,5,5],
          [1,7,5,5],
          [1,2,5,6],
          [1,6,6,6],
          [1,7,6,7]]    # (8, 4)
y_data = [[0,0,1],  # 2 # 이미 원핫인코딩이 되어 있다.
          [0,0,1],
          [0,0,1],
          [0,1,0],  # 1
          [0,1,0],
          [0,1,0],
          [1,0,0],  # 0
          [1,0,0]]  # (8, 3)

x = tf.placeholder('float', [None,4])   # x data의 모양과 맞춰야 한다.
y = tf.placeholder('float', [None,3])   # y daya의 모양과 맞춰야 한다.

# 각 레이어마다 shape을 맞춰줘야 한다.
# x : (None,4) * w : (4,3) >> (None,3) + b : (1,3) >> final shape (None,3) >> hypothesis shape
w = tf.Variable(tf.random_normal([4,3]), name='weight')    # x와 곱할 수 있는 모양 & y와 더할 수 있도록 y의 모양과 맞춰야 한다.
b = tf.Variable(tf.random_normal([1,3]), name='bias')

# 다중분류 : 레이어의 끝을 activation='softmax' 로 감싸준다.
hypothesis = tf.nn.softmax(tf.matmul(x, w) + b) # softmax

# loss = tf.reduce_mean(tf.square(hypothesis-y))  # mse
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1))    # categorical_crossentropy

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

with tf.Session() as sess :
    sess.run(tf.global_variables_initializer())

    for step in range(2001) :
        _, cost_val = sess.run([optimizer, loss], feed_dict={x:x_data, y:y_data})
        if step % 20 == 0 :
            print(step, " / loss : ", cost_val)
    # predict
    a = sess.run(hypothesis, feed_dict={x:[[1, 11, 7, 9]]})
    print("a >> ", a, sess.run(tf.argmax(a,1))) 
    # argmax : a에서 가장 높은 값에 1을 준다.
    # 1999  / loss :  0.4798528
    # a >>  [[0.80384046 0.19088006 0.00527951]] [0] >> 0번째가 1이다.