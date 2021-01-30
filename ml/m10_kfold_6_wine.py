import numpy as np
from sklearn.datasets import load_wine
from sklearn.preprocessing import MinMaxScaler, StandardScaler  # 둘 중에 하나 사용
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier  # Classifier : 분류모델
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression 

# Data
wine = load_wine()
data = wine.data 
target = wine.target 
# print(data.shape)  #(178, 13)
# print(target.shape)  #(178,)

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
# scores :  [0.68       0.84       0.88       0.92       0.91666667]
# <class 'sklearn.svm._classes.SVC'>
# scores :  [0.64 0.8  0.56 0.76 0.75]
# <class 'sklearn.neighbors._classification.KNeighborsClassifier'>
# scores :  [0.64  0.76  0.76  0.56  0.625]
# <class 'sklearn.tree._classes.DecisionTreeClassifier'>
# scores :  [0.96       0.8        0.96       0.84       0.91666667]
# <class 'sklearn.ensemble._forest.RandomForestClassifier'>
# scores :  [1.         0.96       1.         0.96       0.95833333]
# <class 'sklearn.linear_model._logistic.LogisticRegression'>
# scores :  [0.88 0.92 0.96 0.92 1.  ]