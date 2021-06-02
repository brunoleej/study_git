import numpy as np
import tensorflow as tf
from tensorflow.keras.backend import K

EPOCHS = 30
batch_size = 1

x = np.arange(start = 1,stop = 9).astype('float32')
y = np.arange(start = 1,stop = 9).astype('float32')

print(x.shape)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(10,input_shape=(1,)),
    tf.keras.layers.Dense(10),
    tf.keras.layers.Dense(1)
])
def custom_mse(y_true,y_pred):
    return tf.math.reduce_mean(tf.square(y_true,y_pred))

def quantile_loss(y_true,y_pred):
    qs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    q = tf.constant(np.array(qs),dtype = tf.float32)
    e = y_true - y_pred
    v = tf.maximum(q*e, (q-1)*e)
    return K.mean(v)

def quantile_dacon(y_true,y_pred):
    qs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    q = tf.constant(np.array(qs),dtype = tf.float32)
    e = y_true - y_pred
    v = tf.maximum(q*e, (q-1)*e)
    return K.mean(v)

# model
model.compile(loss=quantile_loss,optimizer = 'adam')
model.fit(x,y,epochs=EPOCHS,batch_size=batch_size)

loss = model.evaluate(x,y)

print(loss)