# feature_importances
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

# Data
wine = load_wine()
x_train, x_test, y_train, y_test = \
    train_test_split(wine.data, wine.target, test_size = 0.3, random_state=44)

# Modeling
model = DecisionTreeClassifier(max_depth=4)

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
    n_features = wine.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_,
        align = 'center')
    plt.yticks(np.arange(n_features), wine.feature_names)
    plt.xlabel("Feature Importances")
    plt.ylabel("Features")
    plt.ylim(-1, n_features)    # 축의 한계를 설정한다.

plot_feature_importances_dataset(model)
plt.show()

# feature_importances : 
#  [0.         0.01977277 0.         0.         0.         0.02059664
#  0.18038186 0.         0.         0.         0.06176986 0.30833088
#  0.40914799]
# acc :  0.9259259259259259