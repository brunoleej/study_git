# boston : Regression Problem
# Moudule import 
import numpy as np
import tensorflow as tf
from sklearn.datasets import load_boston

# Data
boston = load_boston()
data = boston.data
target = boston.target

print(data.shape) # (506, 13)
print(target.shape)  # (506, )
# print("=================")
# print(x[:5]) 
# print(y[:10])
# print(np.max(x), np.min(x)) # 711.0  0,0
# print(dataset.feature_names)

# Preprocessing
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data, target, train_size = 0.8, random_state = 66)
print(x_train.shape,x_test.shape)   # (404, 13) (102, 13)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],1)

print(y_train.shape) # (404, 1)

# Modeling
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv1D(32,3,activation='relu',input_shape=(13,1)),
    tf.keras.layers.Conv1D(32,6,activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128,activation = 'relu'),
    tf.keras.layers.Dense(1)
])

# Compile
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mae'])

# Early Stopping
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor = 'loss', patience=20, mode='min') 
# Fit
model.fit(x_train, y_train, batch_size = 32, callbacks=[es], epochs=2000, validation_split=0.2)

# Evaluate
loss,mae = model.evaluate(x_test, y_test, batch_size=32)
print("loss: ", loss)
print('mae: ',mae)

# prediction
y_predict = model.predict(x_test)

# RMSE
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict) : 
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))

# R2_score
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)

# early_stopping (5)
# loss :  10.76313304901123
# mae :  2.4629220962524414
# RMSE :  3.2807214371463127
# R2 :  0.8712281059778333

# early_stopping (10) 
# loss :  8.8392915725708
# mae :  2.440977096557617
# RMSE :  2.973094694598015
# R2 :  0.8942452569241743

# early_stopping (20) 
# loss :  6.976583957672119
# mae :  2.0358965396881104
# RMSE :  2.6413223452327177
# R2 :  0.91653100556001

# CNN model
# loss:  21.7302303314209
# mse:  3.436227560043335
# RMSE:  4.6615693080766
# R2:  0.7145729875723015

# CNN model final output Activation 'linear'
# loss:  21.139278411865234
# mse:  3.452582836151123
# RMSE:  4.597746488205329
# R2:  0.7020289867320698

# CNN model final output Activation 'linear'(second try)
# loss:  21.682680130004883
# mse:  3.6008100509643555
# RMSE:  4.6564665050609255
# R2:  0.7518755394965633

# Conv1D
# loss:  9.405016899108887
# mae:  2.1731979846954346
# RMSE :  3.066760183546865
# R2 :  0.88747681855145