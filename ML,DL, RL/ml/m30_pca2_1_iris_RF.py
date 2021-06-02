# PCA : 차원축소
# RandomForest
import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import r2_score, accuracy_score
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

# Data
iris = load_iris()
data = iris.data
target = iris.target
print(data.shape, target.shape) # (150, 4) (150,)
# print(data.shape[1])

for i in range(data.shape[1]) : 
    i = i + 1
    pca = PCA(n_components=i)
    data2 = pca.fit_transform(data)  

    x_train, x_test, y_train, y_test = train_test_split(data2, target, train_size=0.8, shuffle=True, random_state=46)

    scaler = MinMaxScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    # print(x_train.shape)            # (120, 2) 
    # print(x_test.shape)             # (30, 2) 

    # pca = PCA()
    # pca.fit(x)
    # cumsum = np.cumsum(pca.explained_variance_ratio_)   
    # print("cumsum : ", cumsum) 
    # cumsum :  [0.92461872 0.97768521 0.99478782 1.        ]

    # d = np.argmax(cumsum >= 0.95)+1 # cumsum이 0.95 이상인 컬럼을 True 로 만든다.
    # print("cumsum >= 0.95", cumsum > 0.95)
    # print("d : ", d)
    # cumsum >= 0.95 [False  True  True  True]
    # d :  2

    # import matplotlib.pyplot as plt
    # plt.plot(cumsum)
    # plt.grid()
    # plt.show()
    
    
    # Modeling  
    model = RandomForestClassifier()
    # model = XGBRegressor(n_jobs=-1, use_label_encoder=False)

    # Fitting
    model.fit(x_train, y_train)
    # model.fit(x_train, y_train, eval_metric='logloss')

    # Evaluate
    print("n_components ", i)
    result = model.score(x_test, y_test)
    print("model.score : ", result)

    y_pred = model.predict(x_test)

    score = accuracy_score(y_pred, y_test)
    print("accuracy_score : ", score)

# n_components  1
# model.score :  0.9
# accuracy_score :  0.9
# n_components  2
# model.score :  0.9333333333333333
# accuracy_score :  0.9333333333333333
# n_components  3
# model.score :  0.9
# accuracy_score :  0.9
# n_components  4
# model.score :  0.9333333333333333
# accuracy_score :  0.9333333333333333