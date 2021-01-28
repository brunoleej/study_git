import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler  
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score

# 모델 결과값 비교
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression 

# Data
iris = load_iris()
data = iris.data 
target = iris.target 

# print(data.shape)  # (150, 4)
# print(target.shape)  # (150, )

# Preprocessing 
x_train, x_test, y_train, y_test = train_test_split(data,target , random_state=77, shuffle=True, train_size=0.8)
print(x_train.shape)    # (120, 4)

kfold = KFold(n_splits=5, shuffle=True) # 데이터 5등분

# Modeling
# model = LinearSVC()
# model = SVC()
# model = KNeighborsClassifier()
# model = DecisionTreeClassifier()
# model = RandomForestClassifier()
model = LogisticRegression()

# train, val 5등분
scores = cross_val_score(model, x_train, y_train, cv=kfold)
# print(x_train.shape)    # (120, 4)
print('scores : ', scores)  

# LinearSVC()
# scores :  [1.         0.95833333 0.95833333 1.         0.95833333]

# SVC()
# scores :  [0.95833333 0.83333333 0.95833333 0.95833333 0.95833333]

# KNeighborsClassifier()
# scores :  [1.         0.95833333 1.         1.         1.        ]

# DecisionTreeClassifier()
# scores :  [1.         1.         0.95833333 0.95833333 0.95833333]

# RandomForestClassifier()
# scores :  [1.         0.95833333 0.95833333 0.95833333 1.        ]

# LogisticRegression()
# scores :  [0.95833333 1.         0.91666667 1.         0.91666667]