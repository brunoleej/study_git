# 다중분류 softmax
import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import accuracy_score


tf.set_random_seed(66)

dataset = load_iris()
x_data = dataset.data
y_data = dataset.target
y_data = y_data.reshape(-1,1)
print(x_data.shape, y_data.shape) # (150, 4) (150,1)

# preprocess

x_data, x_test, y_data, y_test = train_test_split(x_data, y_data, train_size=0.8, shuffle=True, random_state=42)

scaler = MinMaxScaler()
scaler.fit(x_data)
x_data = scaler.transform(x_data)
x_test = scaler.transform(x_test)

encoder = OneHotEncoder(categories='auto')
encoder.fit(y_data)
y_data = encoder.transform(y_data).toarray()
y_test = encoder.transform(y_test).toarray()
print(x_data.shape, y_data.shape) # (120, 4) (120, 3)

x = tf.placeholder('float', [None,4])
y = tf.placeholder('float', [None,3])

w = tf.Variable(tf.random_normal([4,3]), name='weight')
b = tf.Variable(tf.random_normal([1,3]), name='bias')

# softmax
hypothesis = tf.nn.softmax(tf.matmul(x, w)+b)
# categorical crossentropy
cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1))   

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.81).minimize(cost)


with tf.Session() as sess :
    sess.run(tf.global_variables_initializer())

    for step in range(2001) :
        _, cost_val = sess.run([optimizer, cost], feed_dict={x:x_data, y:y_data})
        if step % 20 == 0:
            print(step, "/ cost : ", cost_val)

    y_predict = sess.run(hypothesis, feed_dict={x:x_test})
    predicted = sess.run(tf.argmax(y_predict,1))
    test_data = sess.run(tf.argmax(y_test, 1))
    print("accuracy_score : ", accuracy_score(test_data, predicted))

# 2000 / cost :  0.1079638
# accuracy_score :  1.0