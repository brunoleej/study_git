import numpy as np
from sklearn.datasets import load_breast_cancer

from sklearn.preprocessing import MinMaxScaler, StandardScaler  # 둘 중에 하나 사용
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score

# 모델마다 결과 값 비교
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor  # Classifier : 분류모델
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Data
dataset = load_breast_cancer()
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
# model = LinearSVC()
# model = SVC()
# model = KNeighborsClassifier()
# model = DecisionTreeClassifier()
# model = RandomForestClassifier()
model = LogisticRegression()

# fitting
model.fit(x_train, y_train)

# prediction
y_pred = model.predict(x_test)      
print("y_pred : ", y_pred)  

# Evalueate
result = model.score(x_test, y_test)
print("model.score : ", result)         

acc = accuracy_score(y_test, y_pred)
print("accuracy.score : ", acc)      

# LinearSVC
# model.score :  0.9736842105263158
# accuracy.score :  0.9736842105263158

# SVC
# model.score :  0.9736842105263158
# accuracy.score :  0.9736842105263158

# KNeighborClassifier
# model.score :  0.956140350877193
# accuracy.score :  0.956140350877193

# DecisiontreeClassifier
# model.score :  0.9035087719298246
# accuracy.score :  0.9035087719298246

# RandomForestClassifier
# model.score :  0.956140350877193
# accuracy.score :  0.956140350877193

# LogisticRegression
# model.score :  0.9649122807017544
# accuracy.score :  0.9649122807017544