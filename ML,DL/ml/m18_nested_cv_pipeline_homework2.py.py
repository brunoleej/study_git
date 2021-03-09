# 과제
# RandomForest 사용
# 파이프라인 엮어서 25번 돌리기
import numpy as np
from sklearn.datasets import load_wine
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
wine = load_wine()
data = wine.data 
target = wine.target 
# print(data.shape, target.shape)

# preprocessing
# x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=44)
kfold = KFold(n_splits=5, shuffle=True)

# Modeling
# pipline : 파라미터튜닝, 전처리
# # [1] Pipeline
# parameters=[
#     {'model__n_estimators' : [100, 200, 300], 'model__max_depth' : [6, 8, 10, 12]},
#     {'model__max_depth' : [6, 8, 10, 12], 'model__min_samples_leaf' : [3, 7, 10]},
#     {'model__min_samples_split' : [2, 3, 5, 9], 'model__n_jobs' : [-1, 2, 4]}
# ]

# # [2] make_pipeline
# parameters=[
#     {'randomforestclassifier__n_estimators' : [100, 200], 'randomforestclassifier__max_depth' : [6, 8, 10, 12]},
#     {'randomforestclassifier__max_depth' : [6, 8, 10, 12], 'randomforestclassifier__min_samples_leaf' : [3, 7, 10]},
#     {'randomforestclassifier__min_samples_split' : [2, 3, 5, 9], 'randomforestclassifier__n_jobs' : [-1, 2, 4]}
# ]

for train_index, test_index in kfold.split(data) :

    x_train, x_test = data[train_index], data[test_index]
    y_train, y_test = target[train_index], target[test_index]

    model = Pipeline([("scaler", MinMaxScaler()), ('model', RandomForestClassifier())])
    # model = RandomizedSearchCV(pipe, parameters, cv=5)
    score = cross_val_score(model, x_train, y_train, cv=kfold )
    print('교차검증점수 : ', score)

# 교차검증점수 :  [0.96551724 1.         0.96428571 1.         0.96428571]
# 교차검증점수 :  [1.         1.         0.92857143 1.         1.        ]
# 교차검증점수 :  [0.96551724 1.         0.92857143 1.         1.        ]
# 교차검증점수 :  [0.96551724 1.         1.         0.96428571 1.        ]
# 교차검증점수 :  [0.96551724 1.         0.96551724 1.         0.92857143]