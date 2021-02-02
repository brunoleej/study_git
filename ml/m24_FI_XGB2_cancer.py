# xgboosting (pip install xgboost)
# n_jobs 시간 확인 
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import datetime

# Data
cancer = load_breast_cancer()
data = cancer.data 
target = cancer.target

data_df = pd.DataFrame(data, columns=cancer['feature_names']) 
data = data_df.drop(['worst compactness', 'mean symmetry', 'concave points error', 'mean perimeter', 'symmetry error', 'mean compactness', 'worst symmetry', 'mean fractal dimension'], axis=1)
data = data.to_numpy()
# print(data.shape)
# print(target.shape)

x_train, x_test, y_train, y_test = \
    train_test_split(data, target, test_size = 0.3, random_state=44)

start = datetime.datetime.now()

# Modeling
# model = DecisionTreeClassifier(max_depth=4)
# model = RandomForestClassifier()
# model = GradientBoostingClassifier()
model = XGBClassifier(n_jobs = -1, use_label_encoder=False)      # n_jobs = -1 : CPU 자원을 모두 사용하겠다.
# model = XGBClassifier(n_jobs = 1)      

# Fitting
model.fit(x_train, y_train, eval_metric='logloss')

# Evaluate
acc = model.score(x_test, y_test)

print("feature importances : \n", model.feature_importances_)  
print("acc : ", acc) 

end = datetime.datetime.now()
print("time : ", end-start)

# n_jobs 시간 비교
# n_jobs = -1 time :  0:00:00.074800
# n_jobs = 8  time :  0:00:00.069813 
# n_jobs = 4  time :  0:00:00.086740
# n_jobs = 1  time :  0:00:00.101728

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
# ['worst compactness', 'mean symmetry', 'concave points error', 'mean perimeter', 'symmetry error', 'mean compactness', 'worst symmetry', 'mean fractal dimension']
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
#  [0.01097116 0.03097648 0.         0.00770928 0.00120163 0.31298706
#  0.00903356 0.00469326 0.00347319 0.00558912 0.0029177  0.00479884
#  0.00210891 0.0021735  0.07852165 0.02522861 0.362546   0.0162369
#  0.00820302 0.01046769 0.09083616 0.00932618]
# acc :  0.9649122807017544
# time :  0:00:00.053310