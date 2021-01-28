import numpy as np
from sklearn.datasets import load_wine

from sklearn.preprocessing import MinMaxScaler, StandardScaler  # 둘 중에 하나 사용
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score

# 모델마다 나오는 결과 값을 비교한다.
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor  # Classifier : 분류모델
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Data
dataset = load_wine()
data = dataset.data
target = dataset.target 

print(data.shape)  # (178, 13)
print(target.shape)  # (178,)

# Preprocessing
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split (data, target, train_size=0.8, shuffle=True, random_state=66)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# Modeling
# model = LinearSVC()
# model = SVC()
# model = KNeighborsClassifier()
# model = DecisionTreeClassifier()
# model = RandomForestClassifier()
# model = LogisticRegression()

# fitting
model.fit(x_train, y_train)

# Prediction
y_pred = model.predict(x_test)
print("y_pred : ", y_pred)

# Evaluate
result = model.score(x_test, y_test)
print("model.score : ", result)

acc = accuracy_score(y_test, y_pred)
print("accuracy_score : ", acc)


# LinearSVC
# model.score :  0.9722222222222222
# accuracy_score :  0.9722222222222222

# SVC
# model.score :  1.0
# accuracy_score :  1.0

# KNeighborsClassifier
# model.score :  1.0
# accuracy_score :  1.0

# DecisionTreeClassifier
# model.score :  0.9444444444444444
# accuracy_score :  0.9444444444444444

# RandomForestClassifier
# model.score :  1.0
# accuracy_score :  1.0

# LogisticRegression
# model.score :  1.0
# accuracy_score :  1.0