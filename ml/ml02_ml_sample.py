import numpy as np
from sklearn.datasets import load_iris

# Data
iris = load_iris()
x = iris.data
y = iris.target

print(iris.DESCR)
print(iris.feature_names)
print(x.shape,y.shape)
print(x[:5])
print(y)

# Preprocessing
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,train_size = 0.7,shuffle = True)

print(y_train)
print(x_train.shape)  # (105,4)
print(y_train.shape)  # (105,3)

# Model
from sklearn.svm import LinearSVC

model = LinearSVC()
model.fit(x_train,y_train)

result = model.score(x_test,y_test)
print(result)

y_pred = model.predict(x[-5:-1])
print(y_pred)
print(y[-5:-1])