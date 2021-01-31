# train / test / val 
# 중첩 cv
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler  
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression 

import warnings
warnings.filterwarnings('ignore')

# Data
iris = load_iris()
data = iris.data 
target = iris.target 
# print(data.shape)  #(150, 4)
# print(target.shape)  #(150, )

# preprocessing
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=44)

kfold = KFold(n_splits=5, shuffle=True) # 데이터 5등분
parameters = [
    {"C" : [1, 10, 100, 1000], "kernel" : ["linear"]},                              # 4번 계산
    {"C" : [1, 10, 100], "kernel" : ["rbf"], "gamma" : [0.001, 0.0001]},            # 6번 계산
    {"C" : [1, 10, 100, 1000], "kernel" : ["sogmoid"], "gamma" : [0.001, 0.0001]}   # 8번 계산
]   # 한 번 돌 때 18번 파라미터 계산

# Modeling
# Cross_validation
model = GridSearchCV(SVC(), parameters, cv=kfold)          
score = cross_val_score(model, x_train, y_train, cv=kfold)   
print('교차검증점수 : ', score)

# 교차검증점수 :  [0.9047619  0.95238095 0.9047619  0.95238095 0.9047619 ]