# 수동으로 train data, test data, validation data 구분하기 귀찮다.
# train_test_split를 사용해서 나눈다.
# from sklearn.model_selection import train_test_split


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import numpy as np

#1. DATA
x = np.array(range(1,101)) #1부터 100까지
y = np.array(range(1,101)) #101부터 200까지

# x_train = x[:60] # 0번째부터 59번째까지 : 1 ~ 60
# x_val = x[60:80] # 60번째부터 79번째까지 : 61 ~ 80
# x_test = x[80:]  # 81 ~ 100

# y_train = y[:60] # 0번째부터 59번째까지 : 1 ~ 60
# y_val = y[60:80] # 60번째부터 79번째까지 : 61 ~ 80
# y_test = y[80:]  # 81 ~ 100

#train, test 데이터 구분을 train_test_split를 사용해서 나눈다.
# sklearn을 활용한다.
from sklearn.model_selection import train_test_split
x_train , x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True) #x, y 데이터 중 80%를 train으로 준다. / 20% 는 테스트로 준다.
#shuffle=True : 랜덤으로 출력된다.
#shuffle=False : 순서대로 출력된다.

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
model.fit(x_train, y_train, epochs=100)

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
