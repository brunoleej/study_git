import numpy as np
from sklearn.datasets import load_boston
from sklearn.preprocessing import MinMaxScaler, StandardScaler  # 둘 중에 하나 사용
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor  # Regressor : 회귀모델
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# Data
boston = load_boston()
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
# scores :  [0.75616702 0.58679922 0.76111017 0.7214212  0.66438112]
# <class 'sklearn.neighbors._regression.KNeighborsRegressor'>
# scores :  [0.48788193 0.24700448 0.46819728 0.559551   0.58533174]
# <class 'sklearn.tree._classes.DecisionTreeRegressor'>
# scores :  [0.66961727 0.69528409 0.69181159 0.82164776 0.81155536]
# <class 'sklearn.ensemble._forest.RandomForestRegressor'>
# scores :  [0.86176123 0.85507548 0.87269732 0.87309022 0.86794083]