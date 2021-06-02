# PCA : 차원축소
# RandomForest
import numpy as np
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import r2_score, accuracy_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

# Data
wine = load_wine()
data = wine.data
target = wine.target
print(data.shape, target.shape) # (178, 13) (178,)

pca = PCA(n_components=2)
data2 = pca.fit_transform(data)  

x_train, x_test, y_train, y_test = train_test_split(data2, target, test_size=0.3, shuffle=True, random_state=46)

print(x_train.shape)            # (142, 2)   
print(x_test.shape)             # (36, 2)    

# pca = PCA()
# pca.fit(data)
# cumsum = np.cumsum(pca.explained_variance_ratio_)   
# print("cumsum : ", cumsum) 
# cumsum :  [0.99809123 0.99982715 0.99992211 0.99997232 0.99998469 0.99999315
#  0.99999596 0.99999748 0.99999861 0.99999933 0.99999971 0.99999992
#  1.        ]

# d = np.argmax(cumsum >= 0.99)+1
# print("cumsum >= 0.99", cumsum > 0.99)
# print("d : ", d)
# cumsum >= 0.99 [ True  True  True  True  True  True  True  True  True  True  True  True  True]
# d :  1

# import matplotlib.pyplot as plt
# plt.plot(cumsum)
# plt.grid()
# plt.show()

# Modeling
# model = Pipeline([("scaler", MinMaxScaler()),("model",RandomForestRegressor())])
model = Pipeline([("scaler", MinMaxScaler()),("model",XGBClassifier())])

# Fitting
# model.fit(x_train, y_train)
model.fit(x_train, y_train)

# Evaluate
result = model.score(x_test, y_test)
print("model.score : ", result)

y_pred = model.predict(x_test)

score = accuracy_score(y_pred, y_test)
print("accuracy_score : ", score)

# model.score :  0.7222222222222222
# accuracy_score :  0.7222222222222222