# feature_importances
# 중요도가 낮은 컬럼 제거한 후 실행 >> 제거하기 전이랑 결과 유사하다

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

#1. DATA
dataset = load_boston()
x = dataset.data 
y = dataset.target

x_pd = pd.DataFrame(x, columns=dataset['feature_names']) 
x = x_pd.drop(['ZN', 'RAD', 'CHAS', 'INDUS'], axis=1)
x = x.to_numpy()

x_train, x_test, y_train, y_test = \
    train_test_split(x, y, train_size=0.8, random_state=44)

#2. modeling
# model = DecisionTreeRegressor(max_depth=4)
# model = RandomForestRegressor()
model = GradientBoostingRegressor()

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

# print(cut_columns(model.feature_importances_, dataset.feature_names, 4))
# ['ZN', 'RAD', 'CHAS', 'INDUS']

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
#  [0.04556376 0.         0.00288921 0.         0.00572861 0.61396724
#  0.00580415 0.094345   0.         0.         0.01858958 0.00343964
#  0.20967282]
# acc :  0.8120351577921485

# 중요도가 낮은 컬럼 제거한 후
# feature_importances : 
#  [0.04357642 0.61289296 0.00288416 0.09430775 0.02452501 0.00343362
#  0.21838008]
# acc :  0.8315993092000131

# # RandomForestRegressor
# feature_importances : 
#  [0.0344125  0.00098024 0.00566516 0.00139049 0.02280452 0.38030136
#  0.01436384 0.07845415 0.00409196 0.01354601 0.01855543 0.01201266
#  0.41342167]
# acc :  0.8940148005379513

# 중요도 하위 25% 컬럼 제거
# feature_importances : 
#  [0.03877684 0.02540956 0.41190397 0.0159394  0.07366978 0.01695155
#  0.02282993 0.01117619 0.38334279]
# acc :  0.8898374136499382

# # GradientBoostingRegressor
# feature_importances : 
#  [2.66158325e-02 2.89676686e-04 2.41230025e-03 1.13227503e-03
#  3.07642674e-02 3.79378006e-01 8.59627431e-03 1.01517340e-01
#  7.00011209e-04 1.17866111e-02 3.47842926e-02 5.62756601e-03
#  3.96395547e-01]
# acc :  0.8951995257000207

# 중요도 하위 25% 컬럼 제거
# feature_importances : 
#  [0.02879529 0.03157487 0.38086509 0.00792521 0.09851812 0.01268356
#  0.03689758 0.00615203 0.39658827]
# acc :  0.9022234557570641

