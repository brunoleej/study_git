# Variable

import tensorflow as tf

# sess = tf.Session()
sess = tf.compat.v1.Session()

x = tf.Variable([2], dtype=tf.float32, name='test')

# [주의] Variable은 session을 통과시키기 전에 변수를 초기화시켜야 한다. 변수 개수와 상관없이 딱 한 번만 해주면 된다.
# init = tf.global_variables_initializer()
init = tf.compat.v1.global_variables_initializer()
sess.run(init)

print(sess.run(x))  # [2.]
