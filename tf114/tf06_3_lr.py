# y = wx + b
# 데이터를 placeholder 를 사용한다.
# lr 수정, epoch 2000번 보다 적게

import tensorflow as tf

tf.set_random_seed(66)  # random 값을 고정시키기 위함

# x_train = [1,2,3]
# y_train = [3,5,7]

x_train = tf.placeholder(tf.float32, shape=[None])  # shape=[None] : shape이 자유롭다.
y_train = tf.placeholder(tf.float32, shape=[None])

W = tf.Variable(tf.random_normal([1]), name='weight')   # 정규분포에 있는 값 중 랜덤한 값을 넣는다.
b = tf.Variable(tf.random_normal([1]), name='bias')

# sess = tf.Session()
# sess.run(tf.global_variables_initializer()) # 변수 초기화
# print(sess.run(W), sess.run(b)) # [0.06524777] [1.4264158]

# y = wx + b에서 y 값(hypothesis)을 예측한다.
hypothesis = x_train * W + b

# loss 최적화해야 함 GradientDescent (model.compile)
cost = tf.reduce_mean(tf.square(hypothesis - y_train))  # loss == 'mse'

# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
# train = optimizer.minimize(cost)            # optimizer의 cost를 최소로 만들도록 훈련시킨다. >> 최적의 weight를 구한다.
# 아래처럼 한 줄로 줄일 수 있다.
# train = tf.train.GradientDescentOptimizer(learning_rate=0.17413843).minimize(cost)
# train = tf.train.GradientDescentOptimizer(learning_rate=0.173556).minimize(cost)
train = tf.train.GradientDescentOptimizer(learning_rate=0.170838).minimize(cost)


# with문 안으로 Session을 열면 프로그램이 끝날 때 자동으로 session을 닫아준다.
with tf.Session() as sess :                         # sess = tf.Session()
    sess.run(tf.global_variables_initializer())     # sess 초기화

    # for step in range(101) : 
    # for step in range(81) : 
    for step in range(41) : 
        cost_val, W_val, b_val, _ = sess.run([cost, W, b, train], feed_dict={x_train:[1,2,3], y_train:[3,5,7]})       
        # sess.run을 통과한 후 4개 반환값이 나온다. 
        # train에 대한 반환값은 필요가 없다. (train 했다는 의미면 됨)
        if step % 20 == 0 :
            print(step, cost_val, W_val, b_val)   # verbose 처럼 출력시키는 부분

# predict [4], [5,6], [6,7,8] 의 결과를 예측하는 코드 추가
# 예측값을 구하는 hypothesis 식에 예측하고자 하는 숫자를 넣는다.
    print(sess.run(hypothesis, feed_dict={x_train:[4]}))
    print(sess.run(hypothesis, feed_dict={x_train:[5,6]}))
    print(sess.run(hypothesis, feed_dict={x_train:[6,7,8]}))
    print(sess.run(hypothesis, feed_dict={x_train:[1113]}))

# 2000 epochs
# 2000 8.298859e-07 [1.9989444] [1.0023993]
# [8.998177]
# [10.997122 12.996066]
# [12.996066 14.99501  16.993954]
# [2225.8276]

# 500 epochs
# 480 8.337035e-13 [1.9999989] [1.0000021]
# [8.999998]
# [10.999998 12.999997]
# [12.999997 14.999996 16.999996]
# [2226.999]

# 100 epochs - learning_rate=0.17413843
# 100 2.535487e-05 [1.9999996] [1.0047052]
# [9.0047035]
# [11.0047035 13.0047035]
# [13.0047035 15.004703  17.004702 ]
# [2227.0042]

# 80 epochs - learning_rate=0.173556
# 80 0.0001412607 [1.999933] [1.0111697]
# [9.010901]
# [11.010835 13.010767]
# [13.010767 15.0107   17.010633]
# [2226.9368]

# 40 epochs learning_rate=0.170838
# 40 0.004945685 [2.0000095] [1.0633404]
# [9.063378]
# [11.063388 13.063397]
# [13.063397 15.063407 17.063417]
# [2227.0737]

