# xgboosting
# plot_importance
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, plot_importance
import datetime

# Data
cancer = load_breast_cancer()
data = cancer.data 
target = cancer.target

# data_df = pd.DataFrame(data, columns=cancer['feature_names']) 
# data = data_df.drop(['worst compactness', 'mean symmetry', 'concave points error', 'mean perimeter', 'symmetry error', 'mean compactness', 'worst symmetry', 'mean fractal dimension'], axis=1)
# data = data.to_numpy()

# print(data.shape)
# print(target.shape)

x_train, x_test, y_train, y_test = \
    train_test_split(data, target, test_size = 0.3, random_state=44)

start = datetime.datetime.now()

# Modeling
# model = DecisionTreeClassifier(max_depth=4)
# model = RandomForestClassifier()
# model = GradientBoostingClassifier()
model = XGBClassifier(n_jobs = -1)      # n_jobs = -1 : CPU 모두 사용
# model = XGBClassifier(n_jobs = 1)      

# Fitting
model.fit(x_train, y_train)

# Evalute
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

# Graph : 컬럼 중 어떤 것이 가장 중요한 것인지 보여줌
# 중요도가 낮은 컬럼은 제거해도 됨 -> 자원이 절약됨
import matplotlib.pyplot as plt
'''
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

plot_importance(model)
plt.show()

# feature importances : 
#  [0.00675269 0.02949877 0.00258634 0.         0.00892249 0.00212474
#  0.00163637 0.26203424 0.         0.00417205 0.0105376  0.00149657
#  0.00254774 0.00436214 0.00363592 0.00396876 0.0006987  0.00287212
#  0.00237055 0.00193919 0.07018456 0.02327763 0.3904875  0.01487531
#  0.00913892 0.00579641 0.01074649 0.10935875 0.00395197 0.01002543]
# acc :  0.9707602339181286