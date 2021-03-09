# xgboost & randomsearch
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from xgboost import XGBRegressor, plot_importance
from sklearn.metrics import accuracy_score, r2_score
import datetime

# Data
boston = load_boston()
data = boston.data 
target = boston.target

x_train, x_test, y_train, y_test = \
    train_test_split(data, target, test_size=0.3, random_state=44)
kf = KFold(n_splits=5, shuffle=True, random_state=47)

# Modeling
parameters = [
    {"n_estimators":[100, 200, 300], "learning_rate":[0.1, 0.3, 0.001, 0.01], "max_depth":[4, 5, 6]},
    {"n_estimators":[90, 100, 110], "learning_rate":[0.1, 0.01, 0.001], "max_depth":[4, 5, 6], "colsample_bytree":[0.6, 0.9, 1]},
    {"n_estimators":[90, 100], "learning_rate":[0.1, 0.05, 0.001], "max_depth":[4, 5, 6], "colsample_bytree":[0.6, 0.9, 1], "colsample_bylevel" :[0.6, 0.7, 0.9]}
]
# model = GridSearchCV(XGBRegressor(n_jobs = -1, use_label_encoder=False), parameters, cv=kf)
model = RandomizedSearchCV(XGBRegressor(n_jobs = -1, use_label_encoder=False), parameters, cv=kf)

# Fitting
model.fit(x_train, y_train, eval_metric='logloss')

# Evaluate
result = model.score(x_test, y_test)
print("model.score: ",result)

y_pred = model.predict(x_test)

score = r2_score(y_pred, y_test)
print("r2_score: ",score)

# GridSearchCV
# model.score: 0.8975683616079551
# r2_score: 0.879692473590086

# RandomizedSearch
# model.score: 0.8975683616079551
# r2_score: 0.879692473590086