import numpy as np
import pandas as pd 
from sklearn.datasets import load_wine
from sklearn.preprocessing import MinMaxScaler, StandardScaler 
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline, make_pipeline

from sklearn.svm import LinearSVC, SVC
# from sklearn.neighbors import KNeighborsClassifier  
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression 

import warnings
warnings.filterwarnings('ignore')

# Data
wine = load_wine()
data = wine.data 
target = wine.target 
# print(data.shape, target.shape)

# preprocessing
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=44)

# Modeling
# pipline : 파라미터튜닝, 전처리까지 >> 전처리와 모델을 합침

# # [1] Pipeline
                # 전처리 scaler 이름설정        # model  이름설정  
# model = Pipeline([("scaler", MinMaxScaler()), ('malddong', SVC())])
# model = Pipeline([("scaler", StandardScaler()), ('malddong', SVC())])

# # [2] make_pipeline
# model = make_pipeline(MinMaxScaler(), SVC())
model = make_pipeline(StandardScaler(), SVC())

model.fit(x_train, y_train)

results = model.score(x_test, y_test)
print("model.score : ", results)   

# model.score :  1.0