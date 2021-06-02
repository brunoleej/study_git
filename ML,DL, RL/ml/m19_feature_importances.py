# feature_importances
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Data
iris = load_iris()
x_train, x_test, y_train, y_test = \
    train_test_split(iris.data, iris.target, test_size = 0.3, random_state=44)

# Modeling
model = DecisionTreeClassifier(max_depth=4)

# fitting
model.fit(x_train, y_train)

# Evaluate
acc = model.score(x_test, y_test)

print(model.feature_importances_)  # [0.         0.00787229 0.4305627  0.56156501]
print("acc :", acc)   # acc : 0.9333333333333333