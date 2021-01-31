# feature_importances
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

# Data
diabetes = load_diabetes()
x_train, x_test, y_train, y_test = \
    train_test_split(diabetes.data, diabetes.target, test_size = 0.3, random_state=44)

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
    n_features = diabetes.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_,
        align = 'center')
    plt.yticks(np.arange(n_features), diabetes.feature_names)
    plt.xlabel("Feature Importances")
    plt.ylabel("Features")
    plt.ylim(-1, n_features)    

plot_feature_importances_dataset(model)
plt.show()

# feature_importances : 
#  [3.98419876e-02 0.00000000e+00 3.28141694e-01 0.00000000e+00
#  2.33221243e-04 5.74542695e-02 0.00000000e+00 0.00000000e+00
#  5.74328828e-01 0.00000000e+00]
# acc :  0.26786921409473463