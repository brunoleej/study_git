# feature_importances

from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


#1. DATA
dataset = load_iris()
x_train, x_test, y_train, y_test = \
    train_test_split(dataset.data, dataset.target, train_size=0.8, random_state=44)

#2. modeling
model = DecisionTreeClassifier(max_depth=4)

#3. Train
model.fit(x_train, y_train)

#4. Score, Predict
acc = model.score(x_test, y_test)

print(model.feature_importances_)  
# [0.         0.         0.96203388 0.03796612] >> 더하면 1 >> feature(열) 첫번째, 두번째 꺼 빼도 상관없다.

print("acc : ", acc)    # acc :  0.9




