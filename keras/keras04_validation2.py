# validation_split
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# np.array
# array : 행렬

# Data
x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([1,2,3,4,5,6,7,8,9,10])

x_test = array([11,12,13,14,15])
y_test = array([11,12,13,14,15])

x_pred = array([16,17,18]) 

# Modeling
model = Sequential([
    Dense(10,input_dim=1, activation='relu'),
    Dense(5),
    Dense(1),
])

# Compile
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
# Fit
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.2) 
# validation_split : 데이터를 20% 를 쪼개서 사용하겠다. (=10개 중에서 2개만 검증하겠다.) 
# validation data를 따로 지정하지 않아도 됨 << 하이퍼파라미터 튜닝

# Evaluate
results = model.evaluate(x_test, y_test, batch_size=1) #loss = 'mse', metrics='mse' 값이 들어간다
print("results :", results)
# Prediction
y_pred = model.predict(x_pred)
print("y_pred: ", y_pred)