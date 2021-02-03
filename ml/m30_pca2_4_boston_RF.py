# PCA : 차원축소, 컬럼 재구성
# RandomForest로 모델링

import numpy as np
from sklearn.datasets import load_boston
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

#1. DATA
datasets = load_boston()
x = datasets.data
y = datasets.target
# print(x.shape, y.shape) # (442, 10) (442,)

pca = PCA(n_components=3)
x2 = pca.fit_transform(x)  # fit_transform : 전처리 fit과 transform 한꺼번에 한다.

x_train, x_test, y_train, y_test = train_test_split(x2, y, train_size=0.8, shuffle=True, random_state=46)

print(x_train.shape)            # (404, 3) >> 컬럼을 압축시켰다. 컬럼 재구성됨
print(x_test.shape)             # (102, 3) >> 컬럼을 압축시켰다. 컬럼 재구성됨

# pca = PCA(n_components=9)
# x2 = pca.fit_transform(x)  # fit_transform : 전처리 fit과 transform 한꺼번에 한다.

# print(x2)
# print(x2.shape)            # (442, 7) >> 컬럼을 압축시켰다. 컬럼 재구성됨

# pca_EVR = pca.explained_variance_ratio_ # 컬럼이 어느 정도의 변화율을 보여주었는지 보여준다.
# print(pca_EVR)
# print(sum(pca_EVR)) 

# pca = PCA()
# pca.fit(x)
# cumsum = np.cumsum(pca.explained_variance_ratio_)   
# print("cumsum : ", cumsum)  # cumsum 누적 합을 계산
# cumsum :  [0.80582318 0.96887514 0.99022375 0.99718074 0.99848069 0.99920791
#  0.99962696 0.9998755  0.99996089 0.9999917  0.99999835 0.99999992
#  1.        ]

# d = np.argmax(cumsum >= 0.99)+1
# print("cumsum >= 0.95", cumsum > 0.95)
# print("d : ", d)
# cumsum >= 0.95 [False False  True  True  True  True  True  True  True  True  True  True  True]
# d :  3

# import matplotlib.pyplot as plt
# plt.plot(cumsum)
# plt.grid()
# plt.show()


#2. Modeling
model = Pipeline([("scaler", MinMaxScaler()),("model",RandomForestRegressor())])
model = Pipeline([("scaler", MinMaxScaler()),("model",XGBRegressor())])

#3. Train
model.fit(x_train, y_train)

#4. Score, Predict
result = model.score(x_test, y_test)
print("model.score : ", result)

y_pred = model.predict(x_test)

score = r2_score(y_pred, y_test)
print("r2_score : ", score)

# RandomForestRegressor
# model.score :  0.36492135045124197
# r2_score :  0.13812397963846867

# XGBRegressor
# model.score :  0.24805184241717648
# r2_score :  0.13475591268031328
