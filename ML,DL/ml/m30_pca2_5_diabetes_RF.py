# PCA : 차원축소
# RandomForest
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

# Data
diabetes = load_diabetes()
data = diabetes.data
target = diabetes.target
# print(data.shape, target.shape) # (442, 10) (442,)

pca = PCA(n_components=8)
data2 = pca.fit_transform(data)  

x_train, x_test, y_train, y_test = train_test_split(data2, target, test_size=0.3, shuffle=True, random_state=46)

print(x_train.shape)            # (353, 8) 
print(x_test.shape)             # (89, 8) 

# pca = PCA()
# pca.fit(data)
# cumsum = np.cumsum(pca.explained_variance_ratio_)   
# print("cumsum : ", cumsum)  
# cumsum :  [0.40242142 0.55165324 0.67224947 0.76779711 0.83401567 0.89428759
#  0.94794364 0.99131196 0.99914395 1.        ]

# d = np.argmax(cumsum >= 0.95)+1
# print("cumsum >= 0.95", cumsum > 0.95)
# print("d : ", d)
# cumsum >= 0.95 [False False False False False False False  True  True  True]
# d :  8

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

# model.score :  0.32739876705526594
# r2_score :  -0.5172354523413083