import numpy as np
import pandas as pd 
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import MinMaxScaler, StandardScaler  
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline, make_pipeline

# from sklearn.linear_model import LinearRegression
# from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor 
# from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

import warnings
warnings.filterwarnings('ignore')

# Data
diabetes = load_diabetes()
data = diabetes.data 
target = diabetes.target 
# print(data.shape, target.shape)

# preprocessing
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=44)

# Modeling
# pipline : 파라미터튜닝,전처리  >> 전처리와 모델을 합침

# # [1] Pipeline
                # 전처리 scaler 이름설정        # model  이름설정  
# model = Pipeline([("scaler", MinMaxScaler()), ('malddong', RandomForestRegressor())])
# model = Pipeline([("scaler", StandardScaler()), ('malddong',RandomForestRegressor())])

# # [2] make_pipeline
# model = make_pipeline(MinMaxScaler(), RandomForestRegressor())
model = make_pipeline(StandardScaler(), RandomForestRegressor())

model.fit(x_train, y_train)

results = model.score(x_test, y_test)
print("model.score : ", results)   

# model.score :  0.40214938957307533