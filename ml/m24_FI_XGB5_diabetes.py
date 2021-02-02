# xgboosting (pip install xgboost)
# n_jobs 시간 확인 
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import datetime

# Data
diabetes = load_diabetes()
data = diabetes.data 
target = diabetes.target

data_df = pd.DataFrame(data, columns=diabetes['feature_names']) 
data = data_df.drop(['s4', 'sex', 's1'], axis=1)
data = data.to_numpy()

x_train, x_test, y_train, y_test = \
    train_test_split(data, target, test_size = 0.3, random_state=44)

start = datetime.datetime.now()

# Modeling
# model = DecisionTreeRegressor(max_depth=4)
# model = RandomForestRegressor()
# model = GradientBoostingRegressor()
model = XGBRegressor(n_jobs=1, use_label_encoer=False)

# Fitting
model.fit(x_train, y_train, eval_metirc='logloss')

# Evaluate
acc = model.score(x_test, y_test)

print("feature_importances : \n", model.feature_importances_)  
print("acc : ", acc)  

end = datetime.datetime.now()
print("time : ", end-start)

# n_jobs 시간 비교
# n_jobs=-1 time :  0:00:00.128655
# n_jobs=8  time :  0:00:00.128656 
# n_jobs=4  time :  0:00:00.134639
# n_jobs=1  time :  0:00:00.140321

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
# ['sex', 'age', 's1']


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

