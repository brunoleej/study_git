# feature_importances
# 중요도가 낮은 컬럼 제거한 후 실행 >> 제거하기 전이랑 결과 유사하다
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# Data
wine = load_wine()
data = wine.data 
target = wine.target

data_df = pd.DataFrame(data, columns=wine['feature_names']) 
data = data_df.drop(['ash', 'nonflavanoid_phenols', 'malic_acid'], axis=1)
data = data.to_numpy()

x_train, x_test, y_train, y_test = \
    train_test_split(data, target, train_size=0.8, random_state=44)

# Modeling
# model = DecisionTreeClassifier(max_depth=4)
model = RandomForestClassifier()

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
# ['ash', 'nonflavanoid_phenols', 'malic_acid']

'''
# Graph : 컬럼 중 어떤 것이 가장 중요한 것인지 보여줌
# 중요도가 낮은 컬럼은 제거해도 됨 -> 자원이 절약됨
import matp lotlib.pyplot as plt
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
#  [0.14492714 0.01782004 0.02129028 0.05216275 0.19241605 0.023224
#  0.14684229 0.07922395 0.1296244  0.19246909]
# acc :  0.9444444444444444