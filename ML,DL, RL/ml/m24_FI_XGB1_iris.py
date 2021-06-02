# xgboosting (pip install xgboost)
# n_jobs 시간 확인 
import pandas as pd
import numpy as np 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import datetime

# Data
iris = load_iris()
data = iris.data 
target = iris.target

data_df = pd.DataFrame(data, columns=iris['feature_names']) 
data = data_df.drop(['sepal width (cm)'], axis=1)
data = data.to_numpy()

x_train, x_test, y_train, y_test = \
    train_test_split(data, target, test_size = 0.3, random_state=44)

start = datetime.datetime.now()

# Modeling
# model = DecisionTreeClassifier(max_depth=4)
# model = RandomForestClassifier()
# model = GradientBoostingClassifier()
model = XGBClassifier(n_jobs = -1, use_label_encoder=False)      # n_jobs = -1: CPU 모두 사용

# Fitting
model.fit(x_train, y_train, eval_metric='logloss')

# Evaluate
acc = model.score(x_test, y_test)

print(model.feature_importances_)  
print("acc : ", acc)  

end = datetime.datetime.now()
print("time : ", end-start)

# n_jobs 시간 비교
# n_jobs=-1 time :  0:00:00.086739 
# n_jobs=8  time :  0:00:00.095744 
# n_jobs=1  time :  0:00:00.077831 *
# n_jobs=4  time :  0:00:00.091923

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

# print(cut_columns(model.feature_importances_, dataset.feature_names, 1)) # ['sepal width (cm)']
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

# [0.02503096 0.8558802  0.11908882]
# acc :  0.9777777777777777
# time :  0:00:00.070765