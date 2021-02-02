# feature_importances
# 중요도 낮은 컬럼 제거 후 실행 >> 없애기 전이랑 비슷함
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

#1. DATA
diabetes = load_diabetes()
data = diabetes.data 
target = diabetes.target

data_df = pd.DataFrame(data, columns=diabetes['feature_names']) 
data = data_df.drop(['s4', 'sex', 's1'], axis=1)
data = data.to_numpy()

x_train, x_test, y_train, y_test = \
    train_test_split(data, target, train_size=0.8, random_state=44)

# Modeling
# model = DecisionTreeRegressor(max_depth=4)
# model = RandomForestRegressor()
model = GradientBoostingRegressor()

# Fitting
model.fit(x_train, y_train)

# Evaluate
acc = model.score(x_test, y_test)

print("feature_importances : \n", model.feature_importances_)  
print("acc : ", acc)  


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

# print(cut_columns(model.feature_importances_, dataset.feature_names, 3))
# ['sex', 's4', 's1']


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

# feature_importances : 
#  [0.06934022 0.28622906 0.08143741 0.08939048 0.04134665 0.37525511
#  0.05700107]
# acc :  0.3936577846504338