# pca를 통해 1.0 인 것은 몇 개?
# m31로 만든 0.95 이상의 n_componet를 사용하여 dnn 모델 생성 
# cnn과 비교
import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

# Data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
data = np.append(x_train, x_test, axis=0)
data = data.reshape(70000, 28*28)  # 3차원은 PCA에 들어가지 않으므로 2차원으로 바꿔줌.
print(data.shape)  # (70000, 784)

target = np.append(y_train, y_test, axis=0)
print(target.shape)  # (70000,)

# pca = PCA() 
# pca.fit(data)
# cumsum = np.cumsum(pca.explained_variance_ratio_)
# print("cumsum : ", cumsum)

# d = np.argmax(cumsum >= 1.0)+1
# print("cumsum >= 1.0", cumsum > 1.0)
# print("d : ", d)    # d : 713

# import matplotlib.pyplot as plt
# plt.plot(cumsum)
# plt.grid()
# plt.show()

pca = PCA(n_components=713)
data2 = pca.fit_transform(data)

print(data2.shape)     # (70000, 713)

x_train, x_test, y_train, y_test = train_test_split(data2, target, test_size=0.3, shuffle=True, random_state=47)
print(x_train.shape)    # (56000, 713)
print(x_test.shape)     # (14000, 713)

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train.shape)    # (56000, 10)
print(y_test.shape)     # (14000, 10)

# Modeling
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

model = Sequential()
model.add(Dense(196, input_shape=(x_train.shape[1],), activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(84, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(56, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(28, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.summary()

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='acc', patience=5, mode='max')

# Compile
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])

# Fitting
model.fit(x_train, y_train, epochs=150, batch_size=32, validation_split=0.2, callbacks=[es])

# Evaluate 
loss, acc = model.evaluate(x_test, y_test, batch_size=32)
print("loss : ",loss)
print("acc : ", acc)

# Prediction
print("y_test[:10] :",np.argmax(y_test[:10],axis=1))
y_predict = model.predict(x_test[:10])
print("y_pred[:10] :",np.argmax(y_predict,axis=1))  

# CNN
# loss :  0.034563612192869186
# acc :  0.9889000058174133
# y_test[:10] : [7 2 1 0 4 1 4 9 5 9]
# y_pred[:10] : [7 2 1 0 4 1 4 9 5 9]

# DNN
# loss :  0.10550455003976822
# acc :  0.9828000068664551
# y_test[:10] : [7 2 1 0 4 1 4 9 5 9]
# y_pred[:10] : [7 2 1 0 4 1 4 9 5 9]

# PCA(>0.95) - DNN
# loss :  0.09774444252252579
# acc :  0.9767143130302429
# y_test[:10] : [3 1 8 1 6 3 5 4 8 3]
# y_pred[:10] : [3 1 8 1 6 3 5 4 8 3]

# PCA(>1.0) - DNN
# loss :  0.14994649589061737
# acc :  0.9728571176528931
# y_test[:10] : [3 1 8 1 6 3 5 4 8 3]
# y_pred[:10] : [3 1 6 1 6 3 5 4 8 3]
