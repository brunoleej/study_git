# 과제
# RandomForest 사용
# 파이프라인 엮어서 25번 돌리기
# load_diabetes
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import MinMaxScaler, StandardScaler  
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline, make_pipeline

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

import warnings
warnings.filterwarnings('ignore')
import pandas as pd 

# Data
diabetes = load_diabetes()
data = diabetes.data 
target = diabetes.target 
# print(data.shape, target.shape)

# preprocessing 
# x_train, x_test, y_train, y_test = train_test_split(data, data, test_size=0.2, random_state=44)
kfold = KFold(n_splits=5, shuffle=True)

# Modeling
# pipline : 파라미터튜닝,전처리
# # [1] Pipeline
# parameters=[
#     {'model__n_estimators' : [100, 200, 300], 'model__max_depth' : [6, 8, 10, 12]},
#     {'model__max_depth' : [6, 8, 10, 12], 'model__min_samples_leaf' : [3, 7, 10]},
#     {'model__min_samples_split' : [2, 3, 5, 9], 'model__n_jobs' : [-1, 2, 4]}
# ]

# # [2] make_pipeline
# parameters=[
#     {'randomforestregressor__n_estimators' : [100, 200], 'randomforestregressor__max_depth' : [6, 8, 10, 12]},
#     {'randomforestregressor__max_depth' : [6, 8, 10, 12], 'randomforestregressor__min_samples_leaf' : [3, 7, 10]},
#     {'randomforestregressor__min_samples_split' : [2, 3, 5, 9], 'randomforestregressor__n_jobs' : [-1, 2, 4]}
# ]

for train_index, test_index in kfold.split(data) :
    # print(train_index,"\n")
    # print(test_index,"\n")
    x_train, x_test = data[train_index], data[test_index]
    y_train, y_test = target[train_index], target[test_index]

    model = Pipeline([("scaler", MinMaxScaler()), ('model', RandomForestRegressor())])
    # model = RandomizedSearchCV(pipe, parameters, cv=kfold)
    score = cross_val_score(model, x_train, y_train, cv=kfold)

    print('교차검증점수 : ', score, "\n")

# 교차검증점수 :  [0.33557035 0.34290468 0.37829745 0.29659838 0.46314587] 
# 교차검증점수 :  [0.34023568 0.15868751 0.32907845 0.43131454 0.53175365] 
# 교차검증점수 :  [0.41242008 0.42793808 0.49491459 0.50753263 0.23469297] 
# 교차검증점수 :  [0.43590458 0.6612307  0.39814197 0.40827537 0.34335122] 
# 교차검증점수 :  [0.48957068 0.23473532 0.36983679 0.47954381 0.48384656] 

