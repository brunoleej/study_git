# 실습 : RandomizedSearch, GridSearch와 Pipeline를 엮음
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler  
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline, make_pipeline

# from sklearn.svm import LinearSVC, SVC
# from sklearn.neighbors import KNeighborsClassifier  
# from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression 

import warnings
warnings.filterwarnings('ignore')
import pandas as pd 

# Data
iris = load_iris()
data = iris.data 
target = iris.target 
# print(x.shape, y.shape)

# preprocessing >>  K-Fold 
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=44)

# 전처리부분을 안써도 됨
# scaler = MinMaxScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

#2. Modeling
# pipline : 파라미터튜닝에 전처리까지 합친다. >> 전처리와 모델을 합친다.

# # [1] Pipeline
# parameters=[
#     {'model__n_estimators' : [100, 200, 300], 'model__max_depth' : [6, 8, 10, 12]},
#     {'model__max_depth' : [6, 8, 10, 12, 14], 'model__min_samples_leaf' : [3, 7, 10]},
#     {'model__min_samples_split' : [2, 3, 5, 9, 10], 'model__n_jobs' : [-1, 2, 4]}
# ]

# # [2] make_pipeline
parameters=[
    {'randomforestclassifier__n_estimators' : [100, 200, 300], 'randomforestclassifier__max_depth' : [6, 8, 10, 12]},
    {'randomforestclassifier__max_depth' : [6, 8, 10, 12, 14], 'randomforestclassifier__min_samples_leaf' : [3, 7, 10]},
    {'randomforestclassifier__min_samples_split' : [2, 3, 5, 9, 10], 'randomforestclassifier__n_jobs' : [-1, 2, 4]}
]

scaler = [MinMaxScaler(), StandardScaler()]
search = [RandomizedSearchCV, GridSearchCV]

for scale in scaler :
    # pipe = Pipeline([("scaler", scale), ('model', RandomForestClassifier())])
    pipe = make_pipeline(scale, RandomForestClassifier())

    for CV in search :
        model = CV(pipe, parameters, cv=5)
        model.fit(x_train, y_train)
        results = model.score(x_test, y_test)
        print(str(scale),str(CV)+':'+str(results))

# MinMaxScaler() <class 'sklearn.model_selection._search.RandomizedSearchCV'>:0.9333333333333333
# MinMaxScaler() <class 'sklearn.model_selection._search.GridSearchCV'>:0.9333333333333333
# StandardScaler() <class 'sklearn.model_selection._search.RandomizedSearchCV'>:0.9
# StandardScaler() <class 'sklearn.model_selection._search.GridSearchCV'>:0.9333333333333333