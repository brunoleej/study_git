import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler, StandardScaler 
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score

# from sklearn.svm import LinearSVC, SVC
# from sklearn.neighbors import KNeighborsClassifier  # Classifier : 분류모델
# from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression # 회귀가 아닌 분류 모델임

import warnings
warnings.filterwarnings('ignore')

import datetime

# Data
cancer = load_breast_cancer()
data = cancer.data 
target = cancer.target 

# preprocessing  
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=44)

kfold = KFold(n_splits=5, shuffle=True) # 데이터를 5등분

parameters=[
    {'n_estimators' : [100, 200, 300, 400], 'max_depth' : [6, 8, 10], 'min_samples_leaf' : [3, 7, 8, 9, 10]},
    {'max_depth' : [6, 8, 10, 12, 14], 'min_samples_leaf' : [3, 7, 10], 'n_jobs' : [-1, 0, 1]},
    {'min_samples_leaf' : [5, 7, 9, 10], 'min_samples_split' : [5, 10, 15]},
    {'min_samples_split' : [2, 3, 5, 9, 10], 'n_jobs' : [-1, 2, 4]}
]

# Modeling
# model = SVC()
model = GridSearchCV(RandomForestClassifier(), parameters, cv=kfold)
# 모델 : SVC 모델을 GridSearchCV로 쌓음
# parameters : SVC에 들어가 있는 파라미터 값들 (딕셔너리 형태)
# 총 90번 모델이 돌아감

# Fitting
start = datetime.datetime.now()
model.fit(x_train, y_train)
end = datetime.datetime.now()
print("time : ", end - start)   # time :  0:02:18.424284

# Evaluate
print("최적의 매개변수 : ", model.best_estimator_)
#  model.best_estimator_ : 어떤 파라미터가 가장 좋은 값인지 알려줌

# Prediction
y_pred = model.predict(x_test)
print('최종정답률', accuracy_score(y_test, y_pred))

aaa = model.score(x_test, y_test)
print('aaa ', aaa)

# 최적의 매개변수 :  RandomForestClassifier(n_jobs=2)
# 최종정답률 0.9649122807017544
# aaa  0.9649122807017544