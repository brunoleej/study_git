# 변수 선언하는 방법 3가지를 사용해서 hypothesis를 출력한다.

import tensorflow as tf

x = [1,2,3]
W = tf.Variable([0.3], tf.float32)
b = tf.Variable([1.0], tf.float32)

hypothesis = W * x + b

#1. sess.run()
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

print("W : ", sess.run(W))
print("b : ", sess.run(b))

v1 = sess.run(hypothesis)
print("v1 hypothesis : ", v1 )
sess.close()

#2. InteractiveSession()
sess = tf.compat.v1.InteractiveSession()
sess.run(tf.compat.v1.global_variables_initializer())

v2 = hypothesis.eval()
print("v2 hypothesis : ", v2 )
sess.close()

#3. .eval(session=sess)
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

v3 = hypothesis.eval(session=sess)
print("v3 hypothesis : ", v3 )
sess.close()

# W :  [0.3]
# b :  [1.]
# v1 hypothesis :  [1.3       1.6       1.9000001]
# v2 hypothesis :  [1.3       1.6       1.9000001]
# v3 hypothesis :  [1.3       1.6       1.9000001]