# validation 값 추가
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import numpy as np

#1. DATA
x = np.array(range(1,101)) #1부터 100까지
y = np.array(range(1,101)) #101부터 200까지

# sklearn을 활용한다.
from sklearn.model_selection import train_test_split
x_train , x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True) 

print(x_train) 
""" x_train 출력결과 (무작위 출력)
[ 13   7  83  89  38  61   5  55  31  46  97  50  29  17  85  67  37 100
  62  79  25  21  16  40  90  58  70  94  91  27  69  66  99  36  20  51
  59  11  28  57  53  80  81  64   4  18  73  43  56  22  52  26  33  39
  42  41  12  60  98  19]
"""
print(x_train.shape) #(80,) >> 스칼라 60개 , 1차원
print(x_test.shape)  #(20,) 
print(y_train.shape) #(80,)
print(y_test.shape)  #(20,)

#2. Modeling
model = Sequential()
model.add(Dense(100, input_dim=1)) # 기본값 : activation='linear' 
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

#3. Compile, Train
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=100, validation_split=0.2) # validation 16개

#4. Evaluate, Predict
loss, mae = model.evaluate(x_test,y_test)
print("loss : ", loss)
print("mae : ", mae)

y_predict = model.predict(x_test)
print(y_predict)

# shuffle = false
# loss :  0.011699133552610874
# mae :  0.10681991279125214

# shuffle = True >> True일 때 결과가 더 좋다.
# loss :  0.0076343403197824955
# mae :  0.06808438152074814

# validation_split = 0.2 >> validation을 넣었더니 성능이 더 떨어졌다. >> 왜? 훈련량 자체가 적어졌기 때문에
# loss :  0.0492154136300087
# mae :  0.174998477101326

