# xgboosting (pip install xgboost)
# n_jobs 시간 확인 
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import datetime

# Data
wine = load_wine()
data = wine.data 
target = wine.target

data_df = pd.DataFrame(data, columns=wine['feature_names']) 
data = data_df.drop(['alcalinity_of_ash', 'nonflavanoid_phenols', 'proanthocyanins'], axis=1)
data = data.to_numpy()

x_train, x_test, y_train, y_test = \
    train_test_split(data, target, test_size = 0.3, random_state=44)

start = datetime.datetime.now()

# Modeling
# model = DecisionTreeClassifier(max_depth=4)
# model = RandomForestClassifier()
# model = GradientBoostingClassifier()
# model = XGBClassifier(n_jobs=-1)
model = XGBClassifier(n_jobs=1, use_label_encoder=False)

# Fitting
model.fit(x_train, y_train, eval_metric='logloss')

# Evaluate
acc = model.score(x_test, y_test)

print("feature_importances : \n", model.feature_importances_)  
print("acc : ", acc)  

end = datetime.datetime.now()
print("time : ", end-start)

# n_jobs 시간비교
# n_jobs=-1 time :  0:00:00.093721
# n_jobs=8  time :  0:00:00.085770
# n_jobs=4  time :  0:00:00.095715
# n_jobs=1  time :  0:00:00.082799 

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
# ['alcalinity_of_ash', 'nonflavanoid_phenols', 'proanthocyanins']

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
#  [0.14278632 0.03949837 0.01505727 0.0411636  0.01752045 0.11933812
#  0.09202785 0.0271138  0.37560457 0.12988964]
# acc :  0.9629629629629629
# time :  0:00:00.049305