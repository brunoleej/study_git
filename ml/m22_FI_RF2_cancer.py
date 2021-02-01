# feature_importances
# 중요도 낮은 컬럼 제거 후 실행 >> 없애기 전이랑 비슷함
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

# Data
cancer = load_breast_cancer()
data = cancer.data 
target = cancer.target

data_df = pd.DataFrame(data, columns=cancer['feature_names']) 
data = data_df.drop(['symmetry error', 'smoothness error', 'mean fractal dimension', 'fractal dimension error', 'mean symmetry', 'compactness error', 'perimeter error', 'concavity error'], axis=1)
data = data.to_numpy()

# print(data.shape)
# print(target.shape)


x_train, x_test, y_train, y_test = \
    train_test_split(data, target, train_size=0.8, random_state=44)

# Modeling
# model = DecisionTreeClassifier(max_depth=4)
model = RandomForestClassifier()

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
# ['symmetry error', 'smoothness error', 'mean fractal dimension', 'fractal dimension error', 'mean symmetry', 'compactness error', 'perimeter error', 'concavity error'] 

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
    plt.ylim(-1, n_features)   

plot_feature_importances_dataset(model)
plt.show()
'''

# feature importances : 
#  [0.02686191 0.01900057 0.04960355 0.04278017 0.00862692 0.00983993
#  0.07160443 0.12301881 0.01607151 0.00515525 0.04292479 0.00522253
#  0.12153072 0.01819053 0.10403877 0.14619345 0.02308754 0.0183717
#  0.02753649 0.10351985 0.00987241 0.00694817]
# acc :  0.9649122807017544