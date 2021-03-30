# 이진분류 : sigmoid, binary_crossentropy
# hypothesis 할 때 sigmoid로 씌워준다.

import tensorflow as tf

tf.set_random_seed(66)

x_data = [[1,2],[2,3],[3,1],
          [4,3],[5,3],[6,2]]
y_data = [[0],[0],[0],
          [1],[1],[1]]  # 0과 1로 이루어진 이진분류

x = tf.placeholder(tf.float32, shape=[None,2])
y = tf.placeholder(tf.float32, shape=[None,1])

w = tf.Variable(tf.random_normal([2,1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# sigmoid를 사용해서 결과값을 0과 1사이로 바꾼다. (sigmoid)
hypothesis = tf.sigmoid(tf.matmul(x, w) + b)    # activation = sigmoid

# cost = tf.reduce_mean(tf.square(hypothesis - y))    # mse
cost = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis)) # binary_crossenropy

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
# tf.cast :  (hypothesis > 0.5 기준) True이면 1, False이면 0을 출력
# 부동소수점에서 정수형으로 바꾼 경우 소수점 버림을 한다.
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted,y), dtype=tf.float32)) # acc
# tf.equal : 두 변수가 같으면 True >> 예측한 값이 맞는지 틀린지를 알 수 있다.
# Boolean형태인 경우 True이면 1, False이면 0을 출력한다.

with tf.Session() as sess :
    sess.run(tf.global_variables_initializer())

    for step in range(5001) :
        cost_val, _ = sess.run([cost, train], feed_dict={x:x_data, y:y_data})

        if step % 200 == 0 :
            print(step, cost_val)

    h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={x:x_data, y:y_data})
    print("예측값 ", h, "\n원래값 ", c, "\nAccuracy ", a )

'''
예측값  [[0.07802196]
 [0.195242  ]
 [0.48302576]
 [0.7082548 ]
 [0.884784  ]
 [0.9640333 ]]
원래값  [[0.]
 [0.]
 [0.]
 [1.]
 [1.]
 [1.]]
Accuracy  1.0
'''