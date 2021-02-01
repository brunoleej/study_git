# feature_importances
# 중요도가 낮은 컬럼 제거한 후 실행 >> 제거하기 전이랑 결과 유사하다

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

#1. DATA
dataset = load_wine()
x = dataset.data 
y = dataset.target

x_pd = pd.DataFrame(x, columns=dataset['feature_names']) 
x = x_pd.drop(['total_phenols', 'alcalinity_of_ash', 'proanthocyanins'], axis=1)
x = x.to_numpy()

x_train, x_test, y_train, y_test = \
    train_test_split(x, y, train_size=0.8, random_state=44)

#2. modeling
# model = DecisionTreeClassifier(max_depth=4)
# model = RandomForestClassifier()
model = GradientBoostingClassifier()

#3. Train
model.fit(x_train, y_train)

#4. Score, Predict
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

# # DecisionTreeClassifier
# feature_importances : 
#  [0.         0.         0.         0.         0.01723824 0.       
#  0.15955687 0.         0.         0.         0.05577403 0.32933594
#  0.43809492]
# acc :  0.8333333333333334

# 중요도 0인 컬럼 제거한 후
# feature_importances : 
#  [0.01723824 0.15955687 0.         0.05577403 0.32933594 0.43809492]
# acc :  0.8611111111111112

# # RandomForestClassifier
# feature_importances : 
#  [0.12850093 0.02507761 0.01757397 0.02856365 0.02978324 0.05940561
#  0.14125624 0.01078567 0.02434625 0.1548911  0.08333419 0.14943613
#  0.14704543]
# acc :  0.9722222222222222

# 중요도 하위 25% 컬럼 제거
# feature_importances : 
#  [0.13335234 0.02338144 0.01487588 0.04290358 0.18181307 0.02349594
#  0.11693244 0.07098034 0.15362394 0.23864105]
# acc :  0.9444444444444444

# # GradientBoostingClassifier()
# feature_importances : 
#  [5.84999416e-02 3.46943602e-02 1.55876589e-02 2.00244560e-03
#  7.96488236e-03 9.31857302e-08 2.12241158e-01 2.68126521e-03
#  2.09001810e-03 2.29978740e-01 1.78863799e-02 1.24959749e-01
#  2.91413307e-01]
# acc :  0.9166666666666666

# 중요도 하위 25% 컬럼 제거
# feature_importances : 
#  [0.05820696 0.03693478 0.01530449 0.0097578  0.19549742 0.00253011
#  0.22007411 0.02159776 0.14843834 0.29165824]
# acc :  0.9166666666666666

