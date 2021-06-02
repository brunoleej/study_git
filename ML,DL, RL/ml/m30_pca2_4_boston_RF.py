# PCA : 차원축소
# RandomForest
import numpy as np
from sklearn.datasets import load_boston
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

# Data
boston = load_boston()
data = boston.data
target = boston.target
# print(data.shape, target.shape) # (442, 10) (442,)

pca = PCA(n_components=3)
data2 = pca.fit_transform(data)  

x_train, x_test, y_train, y_test = train_test_split(data2, target, test_size=0.3, shuffle=True, random_state=46)

print(x_train.shape)            # (404, 3) 
print(x_test.shape)             # (102, 3) 

# pca = PCA(n_components=9)
# data2 = pca.fit_transform(data) 

# print(data2)
# print(data2.shape)            # (442, 7) 

# pca_EVR = pca.explained_variance_ratio_ 
# print(pca_EVR)
# print(sum(pca_EVR)) 

# pca = PCA()
# pca.fit(data)
# cumsum = np.cumsum(pca.explained_variance_ratio_)   
# print("cumsum : ", cumsum)  
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

# Modeling
model = Pipeline([("scaler", MinMaxScaler()),("model",RandomForestRegressor())])
model = Pipeline([("scaler", MinMaxScaler()),("model",XGBRegressor())])

# Fitting
model.fit(x_train, y_train)

# Evaluate
result = model.score(x_test, y_test)
print("model.score : ", result)

y_pred = model.predict(x_test)

score = r2_score(y_pred, y_test)
print("r2_score : ", score)

# model.score :  0.19680374175373427
# r2_score :  0.00829468541689915