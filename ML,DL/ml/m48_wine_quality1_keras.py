import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, RobustScaler
from sklearn.covariance import EllipticEnvelope
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Conv1D, Flatten, MaxPool1D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from xgboost import XGBClassifier, plot_importance

#1. DATA
df = pd.read_csv('../data/csv/winequality-white.csv', header=0, sep=';')
print(df.shape)   # (4898, 12)
# print(df.head())
'''
   fixed acidity  volatile acidity  citric acid  residual sugar  chlorides  free sulfur dioxide  total sulfur dioxide  density    pH  sulphates  alcohol  quality
0            7.0              0.27         0.36            20.7      0.045                 45.0                 170.0   1.0010  3.00       0.45      8.8        6
1            6.3              0.30         0.34             1.6      0.049                 14.0                 132.0   0.9940  3.30       0.49      9.5        6
2            8.1              0.28         0.40             6.9      0.050                 30.0                  97.0   0.9951  3.26       0.44     10.1        6
'''
# print(df.iloc[:,-1].value_counts())
# 6    2198
# 5    1457
# 7     880
# 8     175
# 4     163
# 3      20
# 9       5

x = df.iloc[:,:-1]
y = df.iloc[:,-1]

print(np.unique(y)) # [3 4 5 6 7 8 9]
print(x.shape, y.shape) # (4898, 11) (4898,)

x = np.array(x)
y = np.array(y)

outlier = EllipticEnvelope(contamination=.1)
outlier.fit(x)
yhat = outlier.predict(x)
mask = yhat != -1

x = x[mask,:]
y = y[mask]
print(x.shape, y.shape)  # (4409, 11) (4409,)


newlist = []
# y 카테고리를 3개로 줄여준다. (카테고리 나누는 기준은 알아서 정할 수 있다.)
for i in list(y) :
    if i <=4 : 
        newlist += [0]
    elif i <=7 :
        newlist += [1]
    else :
        newlist += [2]
y = newlist
print(np.unique(y)) # [0 1 2]

y = np.asarray(y)

import matplotlib.pyplot as plt
# plt.boxplot(x)
# plt.boxplot(y)
# plt.show()


# preprocess
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=42)

# scaler = MinMaxScaler()
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1],1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1],1)
print(x_train.shape, x_test.shape)  # (3527, 11, 1) (882, 11, 1)

y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)

encoder = OneHotEncoder()
encoder.fit(y_train)
y_train = encoder.transform(y_train).toarray()
y_test = encoder.transform(y_test).toarray()

print(y_train.shape, y_test.shape)  # (3527, 2) (882, 2)


#2. Modeling

model = Sequential()
model.add(Conv1D(128, 3,activation='relu',padding='same',input_shape=(11,1)))
model.add(Conv1D(64,3,activation='relu',padding='same'))
model.add(MaxPool1D(3, padding='same'))
model.add(Conv1D(32,3,activation='relu',padding='same'))
model.add(Conv1D(32,3,activation='relu',padding='same'))
model.add(MaxPool1D(3, padding='same'))
model.add(Flatten())
model.add(Dense(32,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(3,activation='softmax'))
model.summary()

#3. Compile, Train
es = EarlyStopping(monitor='val_loss', patience=20, mode='min')
rl = ReduceLROnPlateau(monitor='val_loss', patience=15, factor=0.4,mode='min')
# path = '../data/modelcheckpoint/k86_wine_{val_loss:.4f}.hdf5'
# cp = ModelCheckpoint(path, monitor='val_loss', save_best_only=True, mode='min')

model.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=16, validation_split=0.2, callbacks=[es,rl])#,cp])

#4. Evaluate, Predict
result = model.evaluate(x_test, y_test, batch_size=16)
print("loss ", result[0])
print("acc ", result[1])


# MinMaxScaler
# loss  1.0954968929290771
# acc  0.5653061270713806

# RobustScaler
# loss  1.1940757036209106
# acc  0.6071428656578064

# x outlier 제거 
# loss  1.3548438549041748
# acc  0.5861678123474121

# y컬럼 이진분류
# loss  0.6676238775253296
# acc  0.8480725884437561

# y컬럼 3개로 분류
# loss  0.4916940927505493
# acc  0.9285714030265808