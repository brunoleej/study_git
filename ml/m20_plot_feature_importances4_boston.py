# feature_importances
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

# Data
boston = load_boston()
x_train, x_test, y_train, y_test = \
    train_test_split(boston.data, boston.target, test_size = 0.3, random_state=44)

# Modeling
model = DecisionTreeRegressor(max_depth=4)

# Fitting
model.fit(x_train, y_train)

# Evaluate
acc = model.score(x_test, y_test)

print("feature_importances : \n", model.feature_importances_)  
print("acc : ", acc)  

# Graph : 컬럼 중 어떤 것이 가장 중요한 것인지 보여줌
# 중요도가 낮은 컬럼은 제거해도 됨 -> 그만큼 자원이 절약됨
import matplotlib.pyplot as plt
import numpy as np 

def plot_feature_importances_dataset(model) :
    n_features = boston.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_,
        align = 'center')
    plt.yticks(np.arange(n_features), boston.feature_names)
    plt.xlabel("Feature Importances")
    plt.ylabel("Features")
    plt.ylim(-1, n_features)  

plot_feature_importances_dataset(model)
plt.show()

# feature_importances : 
#  [0.04125185 0.         0.         0.         0.02805571 0.60190039
#  0.00418586 0.08286024 0.         0.         0.02160984 0.0045912
#  0.2155449 ]
# acc :  0.8330453949477281