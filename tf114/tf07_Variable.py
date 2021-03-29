# 변수 선언하는 방법 3가지

import tensorflow as tf
tf.compat.v1.set_random_seed(777)

# [1] 변수 선언 Variable() -> Session() -> sess.run(W)
W = tf.Variable(tf.random.normal([1]), name='weight')

# print(W)
# <tf.Variable 'weight:0' shape=(1,) dtype=float32_ref>

# sess = tf.Session()
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

aaa = sess.run(W)
print("aaa ", aaa)  # [2.2086694]
sess.close()

# [2] InteractiveSession() -> W.eval() : InteractiveSession한 후, eval()을 한다.
# sess = tf.InteractiveSession()
sess = tf.compat.v1.InteractiveSession()
sess.run(tf.compat.v1.global_variables_initializer())

bbb = W.eval()  
print("bbb ", bbb)  # [2.2086694]

# [3] Session() -> W.eval(sess) : eval 안에 sess를 넣는다.
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
ccc = W.eval(session=sess)
print("ccc ", ccc)  # [2.2086694]
