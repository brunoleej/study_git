# 가상환경 확인
# 왼쪽 아래 Python 있는 곳 선택해서 가상환경을 바꿀 수 있다.

import tensorflow as tf
print(tf.__version__)   # base 2.3.1 >> tf114 1.14.0

hello = tf.constant("Hello World")  
# Tensorflow의 자료형
# [1] constant 상수 : 바꾸지 않는 값, 고정값 
# [2] varable : 변수  
# [3] placeholder 
number = tf.constant(123)

print(hello)    
# 그냥 출력시키면 Tensor("Const:0", shape=(), dtype=string) >> 자료형이 출력된다.
print(number)

# Session을 실행시켜야 결과가 출력된다.
sess = tf.Session()
print(sess.run(hello))  
# b'Hello World'
print(sess.run(number))
# 123
