import numpy as np

data = np.load('./samsung_prices.npy')


def split_x(seq, size, col) :
    dataset = []  
    for i in range(len(seq) - size + 1) :
        subset = seq[i:(i+size),0:col].astype('float32')
        dataset.append(subset)
    # print(type(dataset))
    return np.array(dataset)

size = 5
col = 6
dataset = split_x(data,size, col)

#1. DATA
x = dataset[:-1,:,:7]
y = dataset[1:,:1,-1:]

x_pred = dataset[-1:,:,:]

# preprocessing
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8,\
    shuffle=True, random_state=166)
x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, \
    train_size=0.8, shuffle=True, random_state=166)


y_train = y_train.reshape(y_train.shape[0],1)
y_test = y_test.reshape(y_test.shape[0],1)


x_train = x_train.reshape(x_train.shape[0],30)
x_test = x_test.reshape(x_test.shape[0],30)
x_validation = x_validation.reshape(x_validation.shape[0],30)
x_pred = x_pred.reshape(x_pred.shape[0],30)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_validation = scaler.transform(x_validation)
x_pred = scaler.transform(x_pred)

x_train = x_train.reshape(x_train.shape[0],5, 6)
x_test = x_test.reshape(x_test.shape[0],5, 6)
x_validation = x_validation.reshape(x_validation.shape[0], 5, 6)
x_pred= x_pred.reshape(x_pred.shape[0], 5, 6)

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout

#2. Modeling
model = load_model('./next_prices_197-69906.2656.h5')


result = model.evaluate(x_test, y_test, batch_size=5)
print("loss : ", result[0])
print("mae : ", result[1])

y_pred = model.predict(x_test)

from sklearn.metrics import mean_squared_error, r2_score
def RMSE (y_test, y_pred) :
    return np.array(mean_squared_error(y_test, y_pred))
print("RMSE : ", RMSE(y_test, y_pred))

r2 = r2_score(y_test, y_pred)
print("R2 : ", r2)

predict = model.predict(x_pred)
print("1월 14일 삼성주가 예측 : ", predict)

# loss :  86109.234375
# mae :  227.0120086669922
# RMSE :  86109.37
# R2 :  0.9994582343698724
# 1월 14일 삼성주가 예측 :  [[86788.21]]