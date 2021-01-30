import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler, StandardScaler  
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression 

# Data
iris = load_breast_cancer()
data = iris.data 
target = iris.target 
# print(data.shape)  # (569, 30)
# print(target.shape)  # (569,)

# Preprocessing
x_train, x_test, y_train, y_test = train_test_split(data, target, random_state=77, shuffle=True, train_size=0.7)

kfold = KFold(n_splits=5, shuffle=True) 

# Modeling
models = [LinearSVC, SVC, KNeighborsClassifier, DecisionTreeClassifier, RandomForestClassifier, LogisticRegression]

for algorithm in models :  
    model = algorithm()
    scores = cross_val_score(model, x_train, y_train, cv=kfold)
    print(algorithm)
    print('scores : ', scores)  

# <class 'sklearn.svm._classes.LinearSVC'>
# scores :  [0.73626374 0.93406593 0.89010989 0.91208791 0.92307692]
# <class 'sklearn.svm._classes.SVC'>
# scores :  [0.87912088 0.9010989  0.89010989 0.92307692 0.94505495]
# <class 'sklearn.neighbors._classification.KNeighborsClassifier'>
# scores :  [0.96703297 0.92307692 0.94505495 0.91208791 0.87912088]
# <class 'sklearn.tree._classes.DecisionTreeClassifier'>
# scores :  [0.93406593 0.95604396 0.94505495 0.93406593 0.96703297]
# <class 'sklearn.ensemble._forest.RandomForestClassifier'>
# scores :  [0.95604396 0.97802198 0.97802198 0.98901099 0.95604396]
# <class 'sklearn.linear_model._logistic.LogisticRegression'>
# scores :  [0.95604396 0.93406593 0.92307692 0.95604396 0.92307692]