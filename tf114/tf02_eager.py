# 즉시 실행 모드 끄기 (2.0 버전에서 1.0 버전의 코딩이 가능해진다.)

# from tensorflow.python.framework.ops import disable_eager_execution 
import tensorflow as tf

print(tf.executing_eagerly())   # (tf114) False / (base)일 때는 True
# 즉시 실행 모드 : sess.run을 하지 않아도 실행시킬 수 있다. 

tf.compat.v1.disable_eager_execution()  # 즉시 실행 모드를 끈다. (2.0 버전에서 1.0 버전의 코딩이 가능해진다.)
print(tf.executing_eagerly())   # False


print(tf.__version__)   # base 2.3.1

hello = tf.constant("Hello World")  
print(hello)    
# Tensor("Const:0", shape=(), dtype=string)

# Session을 실행시켜야 결과가 출력된다.
# sess = tf.Session()
sess = tf.compat.v1.Session()
print(sess.run(hello))  
# b'Hello World'

'''
즉시 실행모드가 켜져있는 2.3.1 상황에서는 에러가 뜬다.
Traceback (most recent call last):
  File "c:\Study\tf114\tf02_eager.py", line 20, in <module>
    sess = tf.Session()
AttributeError: module 'tensorflow' has no attribute 'Session'
'''
