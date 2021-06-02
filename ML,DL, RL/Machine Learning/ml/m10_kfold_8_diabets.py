import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import MinMaxScaler, StandardScaler  # 둘 중에 하나 사용
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor  # Regressor : 회귀모델
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# Data
boston = load_diabetes()
data = boston.data 
target = boston.target 
# print(data.shape)  #(506, 13)
# print(target.shape)  #(506,)

# Preprocessing
x_train, x_test, y_train, y_test = train_test_split(data, target, random_state=77, shuffle=True, train_size=0.8)

kfold = KFold(n_splits=5, shuffle=True) 

# Modeling
models = [LinearRegression, KNeighborsRegressor, DecisionTreeRegressor, RandomForestRegressor]

for algorithm in models :  
    model = algorithm()
    scores = cross_val_score(model, x_train, y_train, cv=kfold) # r2_score
    print(algorithm)
    print('scores : ', scores)  

# <class 'sklearn.linear_model._base.LinearRegression'>
# scores :  [0.46982172 0.58181109 0.52295197 0.40607037 0.42358058]
# <class 'sklearn.neighbors._regression.KNeighborsRegressor'>
# scores :  [0.37684107 0.22209645 0.45583673 0.41644435 0.35600002]
# <class 'sklearn.tree._classes.DecisionTreeRegressor'>
# scores :  [-0.40351015 -0.1171677  -0.34770927 -0.15884677  0.03177834]
# <class 'sklearn.ensemble._forest.RandomForestRegressor'>
# scores :  [0.43648311 0.35761279 0.48975585 0.26179614 0.40068396]