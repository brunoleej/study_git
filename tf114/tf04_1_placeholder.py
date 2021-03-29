# placeholder

import tensorflow as tf
# 기본적인 노드, 고정값으로 정한다.
node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)
node3 = tf.add(node1 , node2)

sess = tf.Session()

# placeholder : input 같은 개념

# [1] plaveholder할 걸 지정한다.
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

# [2] 연산 
adder_node = a + b

# [3] feed_dict=딕셔너리 형태로 연산할 숫자를 넣는다. 
print(sess.run(adder_node, feed_dict={a:3, b:4.5})) 
# 7.5
print(sess.run(adder_node, feed_dict={a:[1,3], b:[3,4]})) 
# [4. 7.]

add_and_triple = adder_node * 3
print(sess.run(add_and_triple, feed_dict={a:4, b:2}))
# 18.0

