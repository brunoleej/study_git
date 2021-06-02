# CNN으로 구성
import numpy as np
import tensorflow as tf
from sklearn.datasets import load_diabetes

diabets = load_diabetes()
data = diabets.data
target = diabets.target

print(data[:5])
print(target[:10])
print(data.shape, data.shape) # (442, 10) (442,)
# print(np.max(data),np.min(target)) 
# print(diabets.feature_names)
# print(diabets.DESCR)

# Preprocessing
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data, target, train_size = 0.8, random_state = 66)
print(x_train.shape,x_test.shape)   # (353, 10) (89, 10)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(353,10,1)
x_test = x_test.reshape(89,10,1)

# Modeling
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv1D(32,3,padding='SAME',activation='relu',input_shape=(10,1)),
    tf.keras.layers.Conv1D(32,6,activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(1)
])

# Compile
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

# Early Stopping
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor = 'loss', patience = 10, mode = 'min')
model.fit(x_train, y_train,epochs= 2000, validation_split=0.2, callbacks=[early_stopping], batch_size=32)

# Evaluate
loss,mae = model.evaluate(x_test, y_test, batch_size=32)
print('loss: ',loss)
print('mae: ',mae)

# Prediction
y_predict = model.predict(x_test)

from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict) :
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))

from sklearn.metrics import r2_score
r2 = r2_score(y_test,y_predict)
print("R2 : " , r2)

# x_train만 preprocessing
# loss :  3136.124755859375
# mae :  46.16136932373047
# RMSE :  56.00111389680871
# R2 :  0.5167788542278127

# CNN diabets
# loss:  5271.126953125
# mse:  57.096527099609375
# RMSE:  72.60252481119431
# R2:  0.07872804961406843

# Conv1D
# loss:  3220.40966796875
# mae:  46.6273078918457
# RMSE :  56.7486551586473
# R2 :  0.5037920160186857
