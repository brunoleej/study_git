# feature_importances
# 중요도 낮은 컬럼 제거 후 실행 >> 없애기 전이랑 비슷함
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np 

# Data
iris = load_iris()
data = iris.data 
target = iris.target

data_df = pd.DataFrame(data, columns=iris['feature_names']) 
data = data_df.iloc[:,2:]
data = data.to_numpy()

iris = load_iris()
x_train, x_test, y_train, y_test = \
    train_test_split(data, target, test_size = 0.3, random_state=44)

# Modeling
model = DecisionTreeClassifier(max_depth=4)

# Fitting
model.fit(x_train, y_train)

# Evaluate
acc = model.score(x_test, y_test)

print(model.feature_importances_)  
print("acc : ", acc)  

'''
# Graph : 컬럼 중 어떤 것이 가장 중요한 것인지 보여줌
# 중요도가 낮은 컬럼은 제거해도 됨 -> 그만큼 자원이 절약됨
import matplotlib.pyplot as plt
import numpy as np 

def plot_feature_importances_dataset(model) :
    n_features = dataset.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_,
        align = 'center')
    plt.yticks(np.arange(n_features), dataset.feature_names)
    plt.xlabel("Feature Importances")
    plt.ylabel("Features")
    plt.ylim(-1, n_features)    

plot_feature_importances_dataset(model)
plt.show()
'''

# [0.95709212 0.04290788]
# acc :  0.9333333333333333