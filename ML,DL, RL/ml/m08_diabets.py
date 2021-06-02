import numpy as np
from sklearn.datasets import load_diabetes

from sklearn.preprocessing import MinMaxScaler, StandardScaler  # 둘 중에 하나 사용
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score

# 모델 결과 값 비교
# from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor  # Regressor : 회귀모델
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
# from sklearn.linear_model import LogisticRegression   # 이거는 분류모델

# Data
dataset = load_diabetes()
data = dataset.data 
target = dataset.target 

print(data.shape)  # (442, 10)
print(target.shape)  # (442,)

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
model = RandomForestRegressor()

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
# model.score :  0.5063891053505036
# r2_score :  0.5063891053505036

# KNeighborRegression()
# model.score :  0.3741821819765594
# r2_score :  0.3741821819765594

# DecisiontreeRegressor()
# model.score :  -0.17168814111373787
# r2_score :  -0.17168814111373787

# RandomForestRegressor()
# model.score :  0.3667520379609165
# r2_score :  0.3667520379609165