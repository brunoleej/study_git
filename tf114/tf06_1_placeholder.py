# y = wx + b
# 데이터를 placeholder 를 사용한다.

import tensorflow as tf

tf.set_random_seed(66)  # random 값을 고정시키기 위함

# x_train = [1,2,3]
# y_train = [3,5,7]

# x_train과 y_train을 placeholder로 지정한다.
x_train = tf.placeholder(tf.float32, shape=[None])  # shape=[None] : shape이 자유롭다.
y_train = tf.placeholder(tf.float32, shape=[None])

W = tf.Variable(tf.random_normal([1]), name='weight')   # 정규분포에 있는 값 중 랜덤한 값을 넣는다.
b = tf.Variable(tf.random_normal([1]), name='bias')

# y = wx + b에서 y 값(hypothesis)을 예측한다.
hypothesis = x_train * W + b

# loss 최적화해야 함 GradientDescent (model.compile)
cost = tf.reduce_mean(tf.square(hypothesis - y_train))  # loss == 'mse'

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)


# with문 안으로 Session을 열면 프로그램이 끝날 때 자동으로 session을 닫아준다.
with tf.Session() as sess :                         # sess = tf.Session()
    sess.run(tf.global_variables_initializer())     # sess 초기화

    for step in range(2001) :   # epochs=2000
        # sess.run(train)
        # do = sess.run(train, feed_dict={x_train:[1,2,3], y_train:[3,5,7]})  
        cost_val, W_val, b_val, _ = sess.run([cost, W, b, train], feed_dict={x_train:[1,2,3], y_train:[3,5,7]})       
        # sess.run을 통과한 후 4개 반환값이 나온다. 
        # train에 대한 반환값은 필요가 없다. (train 했다는 의미면 됨)
        if step % 20 == 0 :
                        # loss          # weight     # bias
            # print(step, do, sess.run(W), sess.run(b))   # verbose 처럼 출력시키는 부분
            print(step, cost_val, W_val, b_val)   # verbose 처럼 출력시키는 부분


'''
0 11.376854 [0.22876799] [1.4952775]
20 0.25022563 [1.4284163] [1.9631107]
40 0.13584478 [1.561582] [1.9646112]
/
/
/
1960 1.3069816e-05 [1.9958011] [1.0095452]
1980 1.187079e-05 [1.9959984] [1.0090965]
2000 1.0781078e-05 [1.9961864] [1.0086691]
'''
