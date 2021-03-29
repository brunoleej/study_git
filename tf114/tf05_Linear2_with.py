# y = wx + b
# optimizer 한 줄로
# with문으로 session 열고 닫기

import tensorflow as tf

tf.set_random_seed(66)  # random 값을 고정시키기 위함

x_train = [1,2,3]
y_train = [3,5,7]

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
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# sess = tf.Session()
# sess.run(tf.global_variables_initializer()) # 변수 초기화

# for step in range(2001) :   # epochs=2000
#     sess.run(train)         
#     if step % 20 == 0 :
#                     # loss          # weight     # bias
#         print(step, sess.run(cost), sess.run(W), sess.run(b))   # verbose 처럼 출력시키는 부분

# # 세션 닫기 - 세션이 계속 열려있으면 메모리를 차지하기 때문에, 파일 실행이 끝나면 세션을 닫아줘야 한다. 
# >> 귀찮아 자동으로 세션을 열고 닫는 기능 : with 문을 사용한다.
# sess.close()


# with문 안으로 Session을 열면 프로그램이 끝날 때 자동으로 session을 닫아준다.
with tf.Session() as sess :                         # sess = tf.Session()
    sess.run(tf.global_variables_initializer())     # sess 초기화

    for step in range(2001) :   # epochs=2000
        sess.run(train)         
        if step % 20 == 0 :
                        # loss          # weight     # bias
            print(step, sess.run(cost), sess.run(W), sess.run(b))   # verbose 처럼 출력시키는 부분

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
