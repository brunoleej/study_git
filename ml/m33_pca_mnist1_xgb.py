# pca를 통해 0.95 인 것은 몇 개?
# m31로 만든 0.95 이상의 n_componet를 사용하여 xgboost 모델 생성 
# cnn과 비교
import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# Data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

data = np.append(x_train, x_test, axis=0)
data = data.reshape(70000, 28*28)  # 3차원은 PCA에 들어가지 않으므로 2차원으로 바꿔준다.
print(data.shape)  # (70000, 784)
data = data/255.

target = np.append(y_train, y_test, axis=0)
print(target.shape)  # (70000,)

# pca = PCA() 
# pca.fit(data)
# cumsum = np.cumsum(pca.explained_variance_ratio_)
# print("cumsum : ", cumsum)

# d = np.argmax(cumsum >= 0.95)+1
# print("cumsum >= 0.95", cumsum > 0.95)
# print("d : ", d)    # d :  154

# import matplotlib.pyplot as plt
# plt.plot(cumsum)
# plt.grid()
# plt.show()

pca = PCA(n_components=154)
data2 = pca.fit_transform(data)

print(data2.shape)     # (70000, 154)

x_train, x_test, y_train, y_test = train_test_split(data2, target, test_size=0.3, shuffle=True, random_state=47)
print(x_train.shape)    # (56000, 154)
print(x_test.shape)     # (14000, 154)

# from tensorflow.keras.utils import to_categorical
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)
# print(y_train.shape)    # (56000, 10)
# print(y_test.shape)     # (14000, 10)

# Modeling
model = XGBClassifier(n_jobs = 8, use_label_encoder=False)

# Fitting
model.fit(x_train, y_train, eval_metric='logloss')

# Prediction
y_pred = model.predict(x_test) 

# Evaluate
result = model.score(x_test, y_test)
print("result : ", result)

score = accuracy_score(y_pred, y_test)
print("accuracy_score : ", score)


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

# PCA(>0.95) - XGBoost
# result :  0.9644285714285714
# acc score :  0.9644285714285714