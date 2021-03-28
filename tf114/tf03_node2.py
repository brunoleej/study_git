# tensorflow 1.0 사칙연산 만들기

import tensorflow as tf
node1 = tf.constant(2.0)
node2 = tf.constant(3.0)

sess = tf.Session()

# 덧셈
node_add = tf.add(node1, node2)
print("더하기 ", sess.run(node_add))

# 뺄셈
node_subtract = tf.subtract(node1, node2)
print("빼기 ", sess.run(node_subtract))

# 곱셈
node_multiply = tf.multiply(node1, node2)
print("곱하기 ", sess.run(node_multiply))

# 나눗셈
node_divide = tf.divide(node1, node2)
print("나누기 ", sess.run(node_divide))

# 나머지
node_mod = tf.math.mod(node1, node2)
print("나머지 ", sess.run(node_mod))

# 더하기  5.0
# 빼기   -1.0
# 곱하기  6.0
# 나누기  0.6666667
# 나머지  2.0

'''
TensorFlow 연산   축약 연산자   설명
tf.add()   a + b   a와 b를 더함
tf.multiply()   a * b   a와 b를 곱함
tf.subtract()   a - b   a에서 b를 뺌
tf.divide()   a / b   a를 b로 나눔
tf.pow()   a ** b     를 계산
tf.mod()   a % b   a를 b로 나눈 나머지를 구함
tf.logical_and()   a & b   a와 b의 논리곱을 구함. dtype은 반드시 tf.bool이어야 함
tf.greater()   a > b     의 True/False 값을 반환
tf.greater_equal()   a >= b     의 True/False 값을 반환
tf.less_equal()   a <= b     의 True/False 값을 반환
tf.less()   a < b     의 True/False 값을 반환
tf.negative()   -a   a의 반대 부호 값을 반환
tf.logical_not()   ~a   a의 반대의 참거짓을 반환. tf.bool 텐서만 적용 가능
tf.abs()   abs(a)   a의 각 원소의 절대값을 반환
tf.logical_or()   a I b   a와 b의 논리합을 구함. dtype은 반드시 tf.bool이어야 함
'''