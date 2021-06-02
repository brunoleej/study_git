# Pipeline : GridSearchCV와 pipeline을 묶음
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler  
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline, make_pipeline

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression 

import warnings
warnings.filterwarnings('ignore')
import pandas as pd 

# Data
iris = load_iris()
data = iris.data 
target = iris.target 
# print(data.shape, target.shape)

# preprocessing
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=44)
# Minmax Scale을 하지 않아도 됨

# Modeling
# pipline : 파라미터튜닝,전처리 

# # [1] Pipeline
# 파라미터 앞에 모델명의 '모델이름__'가 들어가야 함.
# parameters = [
#     {"mal__C" : [1, 10, 100, 1000], "mal__kernel" : ["linear"]},                                   # 4번 계산
#     {"mal__C" : [1, 10, 100], "mal__kernel" : ["rbf"], "mal__gamma" : [0.001, 0.0001]},            # 6번 계산
#     {"mal__C" : [1, 10, 100, 1000], "mal__kernel" : ["sogmoid"], "mal__gamma" : [0.001, 0.0001]}   # 8번 계산
# ]   # 한 번 kfold를 돌 때마다 총 18번 파라미터 계산함

                # 전처리 scaler 이름설정        # model  이름설정  
# pipe = Pipeline([("scaler", MinMaxScaler()), ('mal', SVC())])
# pipe = Pipeline([("scaler", StandardScaler()), ('mal', SVC())])

# # [2] make_pipeline 
# 파라미터 앞에 모델명의 '소문자__'가 들어가야 함
parameters = [
    {"svc__C" : [1, 10, 100, 1000], "svc__kernel" : ["linear"]},                                   # 4번 계산
    {"svc__C" : [1, 10, 100], "svc__kernel" : ["rbf"], "svc__gamma" : [0.001, 0.0001]},            # 6번 계산
    {"svc__C" : [1, 10, 100, 1000], "svc__kernel" : ["sogmoid"], "svc__gamma" : [0.001, 0.0001]}   # 8번 계산
]   # 한 번 kfold를 돌 때마다 총 18번 파라미터 계산함

pipe = make_pipeline(MinMaxScaler(), SVC())
# pipe = make_pipeline(StandardScaler(), SVC())

                    # pipe : 전처리와 모델이 합쳐진 모델
# model = GridSearchCV(pipe, parameters, cv=5 )
model = RandomizedSearchCV(pipe, parameters, cv=5)

# Fitting
model.fit(x_train, y_train)

# Evaluate
results = model.score(x_test, y_test)
print("model.score : ", results)   

# SCV
# model.score :  1.0

