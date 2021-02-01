# feature_importances
# 중요도 낮은 컬럼 제거 후 실행 >> 없애기 전이랑 비슷함
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# Data
cancer = load_breast_cancer()
data = cancer.data 
target = cancer.target

data_df = pd.DataFrame(data, columns=cancer['feature_names']) 
data = data_df.drop(['mean fractal dimension', 'mean radius', 'worst fractal dimension', 'mean smoothness', 'mean symmetry', 'perimeter error', 'smoothness error', 'concavity error'], axis=1)
data = data.to_numpy()
# print(data.shape)
# print(target.shape)

x_train, x_test, y_train, y_test = \
    train_test_split(data, target, train_size=0.8, random_state=44)

# Modeling
# model = DecisionTreeClassifier(max_depth=4)
# model = RandomForestClassifier()
model = GradientBoostingClassifier()

# Fitting
model.fit(x_train, y_train)

# Evaluate
acc = model.score(x_test, y_test)

print("feature importances : \n", model.feature_importances_)  
print("acc : ", acc) 

# 중요도 낮은 피처
def cut_columns(feature_importances, columns, number):
    temp = []
    for i in feature_importances:
        temp.append(i)
    temp.sort()
    temp=temp[:number]
    result = []
    for j in temp:
        index = feature_importances.tolist().index(j)
        result.append(columns[index])
    return result

# print(cut_columns(model.feature_importances_, dataset.feature_names, 8))
# ['mean fractal dimension', 'mean radius', 'worst fractal dimension', 'mean smoothness', 'mean symmetry', 'perimeter error', 'smoothness error', 'concavity error']

'''
# Graph : 컬럼 중 어떤 것이 가장 중요한 것인지 보여줌
# 중요도가 낮은 컬럼은 제거해도 됨 -> 자원이 절약됨
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
'''

# feature importances : 
#  [9.07730951e-03 1.98515221e-04 8.18049243e-04 2.55198549e-03
#  2.22104277e-03 4.11219038e-01 3.81536670e-03 2.53309140e-03
#  7.82222168e-03 1.09635518e-03 9.69507970e-04 2.23023626e-03
#  2.97492742e-03 7.36564223e-02 6.22454287e-02 2.77677358e-01
#  4.91280715e-02 1.35733194e-02 2.47111398e-03 2.04502127e-02
#  5.13003564e-02 1.97006997e-03]
# acc :  0.9649122807017544