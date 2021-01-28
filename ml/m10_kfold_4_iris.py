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

# print(iris.shape)  #(150, 4)
# print(data.shape)  #(150, )

# Preprocessing
x_train, x_test, y_train, y_test = \
    train_test_split(data, target, random_state=77, shuffle=True, train_size=0.8)

kfold = KFold(n_splits=5, shuffle=True) # 데이터 5등분 


# Modeling
models = [LinearSVC, SVC, KNeighborsClassifier, DecisionTreeClassifier, RandomForestClassifier]

for algorithm in models :  
    model = algorithm()
    scores = cross_val_score(model, x_train, y_train, cv=kfold) # accuracy_score
    print('scores : ', scores , '-' + str(algorithm))  

# model = LinearSVC()
# model = SVC()
# model = KNeighborsClassifier()
# model = DecisionTreeClassifier()
# model = RandomForestClassifier()
# model = LogisticRegression()

# scores :  [1.         0.91666667 0.95833333 1.         1.        ] -<class 'sklearn.svm._classes.LinearSVC'>
# scores :  [1.         1.         0.91666667 0.91666667 0.95833333] -<class 'sklearn.svm._classes.SVC'>
# scores :  [1.         1.         0.91666667 0.95833333 1.        ] -<class 'sklearn.neighbors._classification.KNeighborsClassifier'>
# scores :  [0.95833333 0.95833333 0.95833333 0.91666667 0.95833333] -<class 'sklearn.tree._classes.DecisionTreeClassifier'>
# scores :  [0.95833333 0.95833333 0.95833333 1.         0.91666667] -<class 'sklearn.ensemble._forest.RandomForestClassifier'>