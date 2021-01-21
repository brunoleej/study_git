import numpy as np
import tensorflow as tf

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


# model
model.compile(loss=custom_mse,optimizer = 'adam')
model.fit(x,y,epochs=EPOCHS,batch_size=batch_size)

loss = model.evaluate(x,y)

print(loss)