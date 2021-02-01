# feature_importances
# 중요도가 낮은 컬럼 제거한 후 실행 >> 제거하기 전이랑 결과 유사하다
# feature_importances

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np

#1. DATA
dataset = load_diabetes()
x = dataset.data 
y = dataset.target

x_pd = pd.DataFrame(x, columns=dataset['feature_names']) 
x = x_pd.drop(['sex', 's4', 's1'], axis=1)
x = x.to_numpy()

x_train, x_test, y_train, y_test = \
    train_test_split(x, y, train_size=0.8, random_state=44)

#2. modeling
# model = DecisionTreeRegressor(max_depth=4)
model = RandomForestRegressor()

#3. Train
model.fit(x_train, y_train)

#4. Score, Predict
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
'''

# # DecisionTreeRegressor
# feature_importances : 
#  [0.02991191 0.         0.32054901 0.         0.01831924 0.06062798
#  0.         0.         0.57059185 0.        ]
# acc :  0.31490122539834386

# feature_importances : 
#  [0.04789328 0.3230994  0.04316533 0.58584199]
# acc :  0.3002403937235434

# # RandomForestRegressor
# feature_importances : 
#  [0.07030189 0.01003086 0.2502562  0.07685399 0.0498766  0.06264253
#  0.05161761 0.02060058 0.33629239 0.07152735]
# acc :  0.4211587112783205

# 중요도 하위 25% 컬럼 제거
# feature_importances : 
#  [0.08028076 0.26027649 0.09253146 0.09173989 0.06375254 0.33464406
#  0.07677481]
# acc :  0.375718550580461
