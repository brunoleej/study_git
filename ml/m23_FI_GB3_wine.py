# feature_importances
# 중요도 낮은 컬럼 제거 후 실행 >> 없애기 전이랑 비슷함
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

# Data
wine = load_wine()
data = wine.data 
target = wine.target

data_df = pd.DataFrame(data, columns=wine['feature_names']) 
data = data_df.drop(['total_phenols', 'alcalinity_of_ash', 'proanthocyanins'], axis=1)
data = data.to_numpy()

x_train, x_test, y_train, y_test = \
    train_test_split(data, target, test_size = 0.3, random_state=44)

# Modeling
# model = DecisionTreeClassifier(max_depth=4)
# model = RandomForestClassifier()
model = GradientBoostingClassifier()

# Fitting
model.fit(x_train, y_train)

# Evaluate
acc = model.score(x_test, y_test)

print("feature_importances : \n", model.feature_importances_)  
print("acc : ", acc)  

#  중요도 낮은 피처
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
# ['total_phenols', 'alcalinity_of_ash', 'proanthocyanins']

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

# feature_importances : 
#  [0.23176101 0.01013871 0.01357153 0.03846271 0.17716547 0.00368879
#  0.00643347 0.03548946 0.2291922  0.25409666]
# acc :  0.9259259259259259