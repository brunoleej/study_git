# feature_importances
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# Data
cancer = load_breast_cancer()
x_train, x_test, y_train, y_test = \
    train_test_split(cancer.data, cancer.target, test_size = 0.3, random_state=44)

# Modeling
model = DecisionTreeClassifier(max_depth=4)

# Fitting
model.fit(x_train, y_train)

# Evaluate
acc = model.score(x_test, y_test)

print("feature importances : \n", model.feature_importances_)  
print("acc : ", acc) 

# Graph : 컬럼 중 어떤 것이 가장 중요한 것인지 보여줌
# 중요도가 낮은 컬럼은 제거해도 됨 -> 그만큼 자원이 절약됨
import matplotlib.pyplot as plt
import numpy as np 

def plot_feature_importances_dataset(model) :
    n_features = cancer.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_,
        align = 'center')
    plt.yticks(np.arange(n_features), cancer.feature_names)
    plt.xlabel("Feature Importances")
    plt.ylabel("Features")
    plt.ylim(-1, n_features)   

plot_feature_importances_dataset(model)
plt.show()

# feature importances : 
#  [0.         0.0544576  0.         0.         0.         0.
#  0.00765063 0.         0.         0.         0.         0.
#  0.         0.         0.         0.         0.         0.
#  0.         0.         0.         0.         0.75003446 0.
#  0.0113168  0.         0.         0.17654051 0.         0.        ]
# acc :  0.9181286549707602