import numpy as np
from sklearn.datasets import load_boston

from sklearn.preprocessing import MinMaxScaler, StandardScaler  # 둘 중에 하나 사용
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score

# 모델 결과 값 비교
# from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor   
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
# from sklearn.linear_model import LogisticRegression   

# Data
dataset = load_boston()
data = dataset.data 
target = dataset.target 

print(data.shape)  # (506, 13)
print(target.shape)  # (506,)

# Preprocessing
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data, target, train_size=0.8, shuffle=True, random_state=66)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler()
# scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# Modeling
# model = LinearRegression()
# model = KNeighborsRegressor()
# model = DecisionTreeRegressor()
# model = RandomForestRegressor()

# fitting
model.fit(x_train, y_train)

# Prediction
y_pred = model.predict(x_test)      
print("y_pred : ", y_pred)  

# Evaluate
result = model.score(x_test, y_test)    
print("model.score : ", result)         

r2 = r2_score(y_test, y_pred)
print("r2_score : ", r2)      


# LinearRegression()
# model.score :  0.8111288663608667
# r2_score :  0.8111288663608667

# KNeighborsRegressor()
# model.score :  0.8265307833211177
# r2_score :  0.8265307833211177

# DecisionTreeRegressor()
# model.score :  0.8157530418002419
# r2_score :  0.8157530418002419

# RandomForestRegressor()
# model.score :  0.925403438728989
# r2_score :  0.925403438728989