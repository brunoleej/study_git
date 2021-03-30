# MultiVariable 다항식 계산 
# y = w1x1 + w2x2 + w3x3 + b
# shape을 잘 맞춰야 한다!!!

import tensorflow as tf

tf.set_random_seed(66)

x_data = [[73, 51, 65], 
          [92, 98, 11],
          [89, 31, 33],
          [99, 33, 100],
          [17, 66, 79]]  # (5, 3)
y_data = [[152],
          [185],
          [180],
          [205],
          [142]]    # (5, 1)

x = tf.placeholder(tf.float32, shape=[None,3])
y = tf.placeholder(tf.float32, shape=[None,1])

w = tf.Variable(tf.random_normal([3,1]), name='weight') 
# x의 shape(5,3)와 행렬 곱할 수 있는 shape으로 맞춰줘야 한다. >> (3,None)
b = tf.Variable(tf.random_normal([1]), name='bias') 
# y_data(5,1) 와 shape를 맞추는 형태를 갖춰야 한다. bias는 한 개의 값을 갖는다.  >> (1)

# hypothesis = x * w + b
hypothesis = tf.matmul(x, w) + b    # tf.matmul : matrics multiply 행렬간의 곱셈을 할 때 사용

cost = tf.reduce_mean(tf.square(hypothesis - y))

train = tf.train.GradientDescentOptimizer(learning_rate=1e-5).minimize(cost)    # learning rate = 0.0001
# train = tf.train.AdamOptimizer(learning_rate=0.00001).minimize(cost)


with tf.Session() as sess :
    sess.run(tf.global_variables_initializer())

    for step in range(2001) :
        _, cost_val, hyp_val = sess.run([train, cost, hypothesis], feed_dict={x:x_data, y:y_data})
        if step % 20 == 0 :
            print(step, '\n', "cost ", cost_val, '\n', "hypothesis\n", hyp_val)


'''
0 
 25807.014
 [[ 24.971392 ]
 [-29.10679  ]
 [ -4.5602245]
 [ 46.44747  ]
 [ 53.367573 ]]
20
 748.02673
 [[175.53491]
 [159.91158]
 [136.4593 ]
 [226.60089]
 [155.94351]]
~
~
~
1980
 296.03598
 [[171.27554]
 [191.23407]
 [151.92091]
 [212.78763]
 [127.14444]]
2000 
 296.03317
 [[171.27557]
 [191.23401]
 [151.92111]
 [212.78754]
 [127.14451]]
'''

