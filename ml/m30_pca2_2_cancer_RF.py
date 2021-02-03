# PCA : 차원축소
# RandomForest
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import r2_score, accuracy_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

# Data
cancer = load_breast_cancer()
data = cancer.data
target = cancer.target
print(data.shape, target.shape) # (569, 30) (569,)

pca = PCA(n_components=2)
data2 = pca.fit_transform(data)  

x_train, x_test, y_train, y_test = train_test_split(data2, target, test_size=0.3, shuffle=True, random_state=46)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(x_train.shape)            # (455, 1) 
print(x_test.shape)             # (114, 1) 


# pca = PCA()
# pca.fit(data)
# cumsum = np.cumsum(pca.explained_variance_ratio_)   
# print("cumsum : ", cumsum)  
# cumsum :  [0.98204467 0.99822116 0.99977867 0.9998996  0.99998788 0.99999453
#  0.99999854 0.99999936 0.99999971 0.99999989 0.99999996 0.99999998
#  0.99999999 0.99999999 1.         1.         1.         1.
#  1.         1.         1.         1.         1.         1.
#  1.         1.         1.         1.         1.         1.        ]

# d = np.argmax(cumsum >= 0.99)+1
# print("cumsum >= 0.99", cumsum > 0.99)
# print("d : ", d)
# cumsum >= 0.99 [False  True  True  True  True  True  True  True  True  True  True  True
#   True  True  True  True  True  True  True  True  True  True  True  True
#   True  True  True  True  True  True]
# d :  2

# import matplotlib.pyplot as plt
# plt.plot(cumsum)
# plt.grid()
# plt.show()


# Modeling
# model = RandomForestClassifier()
model = XGBClassifier(n_jobs = -1, use_label_encoder=False)

# Fitting
model.fit(x_train, y_train, eval_metric='logloss')

# Evaluate
result = model.score(x_test, y_test)
print("model.score : ", result)

y_pred = model.predict(x_test)

score = accuracy_score(y_pred, y_test)
print("accuracy_score : ", score)

# model.score :  0.9473684210526315
# accuracy_score :  0.9473684210526315