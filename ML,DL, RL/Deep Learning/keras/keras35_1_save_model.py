# module import
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM

# model
model = Sequential([
    LSTM(200,input_shape=(4,1)),
    Dense(100),
    Dense(50),
    Dense(20),
    Dense(10),
    Dense(1)
])

# summary
model.summary()

# model save
model.save('./model/save_keras35.h5')
model.save('.//model//save_keras35_1.h5')
model.save('.\model\save_keras35_2.h5')
model.save('.\\model\\save_keras35_3.h5')

