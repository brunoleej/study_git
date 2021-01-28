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
data = dataset.data 
target = dataset.target 

# print(x.shape)  #(150, 4)
# print(y.shape)  #(150, )

# Preprocessing
kfold = KFold(n_splits=5, shuffle=True) # 데이터 5등분

# Modeling
model = LinearSVC()
# model = SVC()
# model = KNeighborsClassifier()
# model = DecisionTreeClassifier()
# model = RandomForestClassifier()
# model = LogisticRegression()

scores = cross_val_score(model, data, target, cv=kfold) 
print('scores : ', scores)  