import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler 
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score

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
kfold = KFold(n_splits=5, shuffle=True) # 데이터 5등분

# Modeling
# model = LinearSVC()
# model = SVC()
# model = KNeighborsClassifier()
# model = DecisionTreeClassifier()
# model = RandomForestClassifier()
model = LogisticRegression()

scores = cross_val_score(model, data, target, cv=kfold) 
print('scores : ', scores)  

# LinearSVC()
# scores :  [0.96666667 0.93333333 0.93333333 1.         0.96666667]

# SVC()
# scores :  [0.96666667 1.         0.86666667 0.93333333 1.        ]

# KNeighborsClassifier()
# scores :  [0.93333333 0.96666667 0.96666667 0.96666667 1.        ]

# DecisionTreeClassifier()
# scores :  [0.9        1.         1.         0.93333333 0.96666667]

# RandomForestClassifier()
# scores :  [0.96666667 0.96666667 0.86666667 0.9        1.        ]

# LogisticRegression()
# scores :  [0.9        1.         0.96666667 1.         0.96666667]