#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('tensorflow_version', '1.x')


# In[ ]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# For reproducibility
np.random.seed(42)


# In[ ]:


# inputs
x = np.linspace(0, 1, 100, dtype=np.float32)

# ground truth
#slopes = np.random.normal(1, 0.5, 100).astype(np.float32)
#intercept = 2.

slopes = 1
intercept = np.random.normal(2, 0.2, 100).astype(np.float32)

# outputs
y = x * slopes + intercept


# In[ ]:


slopes


# In[ ]:


intercept


# In[ ]:


plt.scatter(x, y)
plt.plot(x, x * 1 + 2., label="ground truth", c="r")
plt.legend()
plt.show()


# In[ ]:


x.dtype


# In[ ]:


y.dtype


# In[ ]:


x.shape


# In[ ]:


y.shape


# In[ ]:


# Inputs
x_pl = tf.placeholder(tf.float32, [100,], name="x_pl")
y_pl = tf.placeholder(tf.float32, [100,], name="y_pl")


# In[ ]:


# Computation
## Variables = Parameters = Weights
w = tf.Variable(.1, tf.float32)
b = tf.Variable(0., tf.float32)


# In[ ]:


## prediction = y_hat = hypothesis
preds = x_pl * w + b # (100,)


# In[ ]:


# objective = loss = cost
loss = tf.reduce_mean(tf.square(preds - y_pl)) # L2 loss


# In[ ]:


# Optimization = Training
optimizer = tf.train.GradientDescentOptimizer(learning_rate=.1)
train_op = optimizer.minimize(loss)


# In[ ]:


# start a session
sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth =True)))


# In[ ]:


# initialize all variables
sess.run(tf.global_variables_initializer())


# In[ ]:


loss_list, w_list, b_list = [], [], []
for epoch in range(20):
    _, _loss, _w, _b = sess.run([train_op, loss, w, b], feed_dict={x_pl:x, y_pl:y})
    print(epoch+1, "\t", _loss, "\t", _w, "\t", _b)
    loss_list.append(_loss)
    w_list.append(_w)
    b_list.append(_b)


# In[ ]:


plt.plot(loss_list, label="loss")
plt.plot(w_list, label="w")
plt.plot(b_list, label="b")
plt.legend()
plt.show()


# In[ ]:


plt.scatter(x, y)
plt.plot(x, x * w_list[-1] + b_list[-1], label="model", c="r")
plt.plot(x, x * 1 + 2., label="ground truth", c="g")
plt.legend()
plt.show()


# In[ ]:




