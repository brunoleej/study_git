# feature_importances

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split


#1. DATA
dataset = load_boston()
x_train, x_test, y_train, y_test = \
    train_test_split(dataset.data, dataset.target, train_size=0.8, random_state=44)

#2. modeling
model = DecisionTreeRegressor(max_depth=4)

#3. Train
model.fit(x_train, y_train)

#4. Score, Predict
acc = model.score(x_test, y_test)

print("feature_importances : \n", model.feature_importances_)  
print("acc : ", acc)  

# Graph : 컬럼 중 어떤 것이 가장 중요한 것인지 보여준다.
# 중요도가 낮은 컬럼은 제거해도 된다. >> 그만큼 자원이 절약된다.
import matplotlib.pyplot as plt
import numpy as np 

def plot_feature_importances_dataset(model) :
    n_features = dataset.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_,
        align = 'center')
    plt.yticks(np.arange(n_features), dataset.feature_names)
    plt.xlabel("Feature Importances")
    plt.ylabel("Features")
    plt.ylim(-1, n_features)    # 축의 한계를 설정한다.

plot_feature_importances_dataset(model)
plt.show()

# feature_importances : 
#  [0.04556376 0.         0.00288921 0.         0.00572861 0.61396724
#  0.00580415 0.094345   0.         0.         0.01858958 0.00343964
#  0.20967282]
# acc :  0.8120351577921485