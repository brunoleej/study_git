import tensorflow as tf
import numpy as np

tf.set_random_seed(66)

dataset = np.loadtxt('../data/csv/data-01-test-score.csv', delimiter=',')
print(dataset)
print(dataset.shape)    # (25, 4)

x_data = dataset[:,:-1]
# y_data = dataset[:,-1].reshape(-1,1)
# y_data = dataset[:,-1:]
y_data = dataset[:, [-1]]
print(x_data.shape, y_data.shape) # (25, 3) (25, 1)

x = tf.placeholder(tf.float32, shape=[None, 3])
y = tf.placeholder(tf.float32, shape=[None, 1])

w = tf.Variable(tf.random_normal([3,1]), name='weight')
# x_data shape (25,3)에 곱할 수 있도록 weight shape 을 맞춘다. >> (3,1)  
b = tf.Variable(tf.random_normal([1]), name='bias')
# y_data shape 과 맞도록 b의 shape을 맞춘다.

hypothesis = tf.matmul(x, w) + b

cost = tf.reduce_mean(tf.square(hypothesis - y))

train = tf.train.GradientDescentOptimizer(learning_rate=1e-5).minimize(cost)

with tf.Session() as sess :
    sess.run(tf.global_variables_initializer())

    for step in range(2001) :
        _, cost_val, hyp_val = sess.run([train, cost, hypothesis], feed_dict={x:x_data, y:y_data})
        if step % 20 == 0 :
            print(step, "\n", "cost ", cost_val)#, "\nhypothesis", hyp_val)

    print("======== predict ========")
    print("[73,80,75] >> ", sess.run(hypothesis, feed_dict={x:[[73.,80.,75.]]}))
    print("[93,88,93] >> ", sess.run(hypothesis, feed_dict={x:[[93.,88.,93.]]}))
    print("[89,91,90] >> ", sess.run(hypothesis, feed_dict={x:[[89.,91.,90.]]}))
    print("[96,98,100] >> ", sess.run(hypothesis, feed_dict={x:[[96.,98.,100.]]}))
    print("[73,66,70] >> ", sess.run(hypothesis, feed_dict={x:[[73.,66.,70.]]}))

# [73,80,75] >>  [[152.83249]]
# [93,88,93] >>  [[184.37338]]
# [93,88,93] >>  [[184.37338]]
# [89,91,90] >>  [[181.15134]]
# [96,98,100] >>  [[198.89235]]
# [73,66,70] >>  [[139.52919]]
