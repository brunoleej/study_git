# y = wx + b

import tensorflow as tf

tf.set_random_seed(66)  # random 값을 고정시키기 위함

x_train = [1,2,3]
y_train = [3,5,7]

W = tf.Variable(tf.random_normal([1]), name='weight')   # 정규분포에 있는 값 중 랜덤한 값을 넣는다.
b = tf.Variable(tf.random_normal([1]), name='bias')

sess = tf.Session()
sess.run(tf.global_variables_initializer()) # 변수 초기화
print(sess.run(W), sess.run(b)) # [0.06524777] [1.4264158]

# y = wx + b에서 y 값(hypothesis)을 예측한다.
hypothesis = x_train * W + b

# loss 최적화해야 함 Gradient Descent
cost = tf.reduce_mean(tf.square(hypothesis - y_train))  # loss == 'mse'

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)            # optimizer의 cost를 최소로 만들도록 훈련시킨다. >> 최적의 weight를 구한다.

# sess = tf.Session()
# sess.run(tf.global_variables_initializer()) # 변수 초기화

# for step in range(2001) :   # epochs=2000
for step in range(4) :   # epochs=2000
    sess.run(train)         
    # if step % 20 == 0 :
                    # loss          # weight     # bias
        # print(step, sess.run(cost), sess.run(W), sess.run(b))   # verbose 처럼 출력시키는 부분
    # print(step, sess.run(cost), sess.run(W), sess.run(b))   # verbose 처럼 출력시키는 부분
    

    print("epoch : ",step)
    print("x : ",x_train)
    print("W : ",sess.run(W))
    print("b : ",sess.run(b))
    print("W*x + b = hypothesis , ",sess.run(hypothesis))
    print("y : ",y_train)
    print("hypothesis - y_train : ",sess.run(hypothesis - y_train))
    print("cost : ",sess.run(cost))
    print("\n\n")

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

'''
# 과제 : Weight가 왜 갱신되었나, epoch=3번을 직접 손으로 계산해본다.
epoch :  0
x :  [1, 2, 3]
W :  [0.22876799]
b :  [1.4952775]
W*x + b = hypothesis ,  [1.7240455 1.9528135 2.1815815]
y :  [3, 5, 7]
hypothesis - y_train :  [-1.2759545 -3.0471864 -4.8184185]
cost :  11.376854

epoch :  1
x :  [1, 2, 3]
W :  [0.37427187]
b :  [1.5562212]
W*x + b = hypothesis ,  [1.9304931 2.304765  2.6790369]
y :  [3, 5, 7]
hypothesis - y_train :  [-1.0695069 -2.695235  -4.320963 ]
cost :  9.026286

epoch :  2
x :  [1, 2, 3]
W :  [0.50375766]
b :  [1.6101259]
W*x + b = hypothesis ,  [2.1138835 2.6176412 3.121399 ]
y :  [3, 5, 7]
hypothesis - y_train :  [-0.8861165 -2.3823588 -3.878601 ]
cost :  7.1681275

epoch :  3
x :  [1, 2, 3]
W :  [0.6190019]
b :  [1.657773]
W*x + b = hypothesis ,  [2.276775  2.8957767 3.5147789]
y :  [3, 5, 7]
hypothesis - y_train :  [-0.7232251 -2.1042233 -3.4852211]
cost :  5.699192
'''
