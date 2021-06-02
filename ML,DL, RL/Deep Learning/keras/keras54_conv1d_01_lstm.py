# 코딩하시오 Conv1D
# result : 80
import numpy as np
import tensorflow as tf

# Data
x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
              [5,6,7],[6,7,8],[7,8,9],[8,9,10],
              [9,10,11],[10,11,12],
              [20,30,40],[30,40,50],[40,50,60]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x_pred = np.array([50,60,70])

print(x.shape,y.shape) # (13, 3) (13,)
x = x.reshape((13,3,1)).astype('float32')
print(x)
print(x.shape)

# Model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv1D(2,2,padding = 'SAME',input_shape = (3,1)),
    tf.keras.layers.LSTM(30,activation = 'relu'),
    tf.keras.layers.Dense(30),
    tf.keras.layers.Dense(20),
    tf.keras.layers.Dense(5),
    tf.keras.layers.Dense(1)
])
# print(model.summary())

# Earlystopping
early_stopping = tf.keras.callbacks.EarlyStopping(monitor = 'loss',patience=20,mode = 'auto')

# Compile, Train
model.compile(loss = 'mse',optimizer = 'adam')
model.fit(x,y,epochs = 500,batch_size = 1,callbacks=[early_stopping])

# x_pred.reshape
x_pred = x_pred.reshape(1,3,1).astype('float32')
print(x_pred.shape)

# evaluate, predict
loss = model.evaluate(x,y)
print('loss: ',loss)

y_pred = model.predict(x_pred)
print('y_pred: ',y_pred)

# loss:  0.05855182558298111
# y_pred:  [[80.57456]]

# Conv1D
# loss:  0.3846977949142456
# y_pred:  [[80.82802]]