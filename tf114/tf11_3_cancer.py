# 이진분류
# 최종 sklearn의 acc_score 값으로 결론 낼 것! 

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, accuracy_score
import tensorflow as tf

tf.set_random_seed(66)

dataset = load_breast_cancer()
x_data = dataset.data
y_data = dataset.target
y_data = y_data.reshape(-1,1)
print(x_data.shape, y_data.shape) # (569, 30) (569, 1)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=0.9, shuffle=True, random_state=42)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x = tf.placeholder(tf.float32, shape=[None, 30])
y = tf.placeholder(tf.float32, shape=[None, 1])

w = tf.Variable(tf.zeros([30,1]), name='weight')    # [30,1] 형태 안에 0을 채운다.
b = tf.Variable(tf.zeros([1]), name='bias')

hypothesis = tf.sigmoid(tf.matmul(x,w) + b) # loss = sigmoid

cost = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis)) # binary_crossentropy

train = tf.train.AdamOptimizer(learning_rate=117e-6).minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))

with tf.Session() as sess : 
    sess.run(tf.global_variables_initializer())

    for step in range(501) :
        cost_val, _ = sess.run([cost, train], feed_dict={x:x_train, y:y_train})
        if step % 200 == 0 :
            print(step, cost_val)
    
    h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={x:x_test, y:y_test})
    print("예측값 ", h, "\n원래값 ", c, "\nAccuracy ", a )

    # y_predict = sess.run(hypothesis, feed_dict={x:x_test})
    # 0,1 로 이루어진 predicted 예측값과 실제 y값으로 accuracy score를 구한다.
    print("accuracy_score : ", accuracy_score(y_test, sess.run(predicted, feed_dict={x:x_test})))   

# Accuracy  0.9298246
# accuracy_score :  0.9298245614035088  (그냥 Accuracy 구한 것과 동일한 값이 나온다.)