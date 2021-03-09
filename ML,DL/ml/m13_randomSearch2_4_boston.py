import numpy as np
from sklearn.datasets import load_boston
from sklearn.preprocessing import MinMaxScaler, StandardScaler  
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, r2_score

# from sklearn.linear_model import LinearRegression
# from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor  
# from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

import warnings
warnings.filterwarnings('ignore')

import datetime 

# Data
boston = load_boston()
data = boston.data 
target = boston.target 

# preprocessing
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=44)

kfold = KFold(n_splits=5, shuffle=True) # 데이터 5등분

parameters = [
    {'n_estimators' : [100, 200, 300], 'max_depth' : [6, 8], 'n_jobs' : [-1, 2, 4]},
    {'max_depth' : [6, 8, 10, 12], 'min_samples_leaf' : [3, 7]},
    {'min_samples_leaf' : [3,5,7,9], 'min_samples_split' : [2, 5]},
    {'min_samples_split' : [2,3,5,7], 'n_jobs' : [-1, 2, 4]}
] 

# Modeling
# model = SVC()
model = RandomizedSearchCV(RandomForestRegressor(), parameters, cv=kfold)
# 모델 : SVC 모델을 GridSearchCV로 쌓음
# parameters : SVC에 들어가 있는 파라미터 값들 (딕셔너리 형태)
# 총 90번 모델이 돌아감

# Fitting
start = datetime.datetime.now()
model.fit(x_train, y_train)
end = datetime.datetime.now()
print("time : ", end - start)   # time :  0:00:11.355361

# Evaluate
print("최적의 매개변수 : ", model.best_estimator_)
#  model.best_estimator_ : 어떤 파라미터가 가장 좋은 값인지 알려줌

y_pred = model.predict(x_test)
print('최종정답률', r2_score(y_test, y_pred))

aaa = model.score(x_test, y_test)
print('aaa ', aaa)

# 최적의 매개변수 :  RandomForestRegressor(max_depth=8, n_estimators=300, n_jobs=-1)
# 최종정답률 0.8867971595534571
# aaa  0.8867971595534572