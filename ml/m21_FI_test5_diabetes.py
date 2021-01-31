# feature_importances
# 중요도가 낮은 컬럼 제거한 후 실행 >> 제거하기 전이랑 결과 유사하다
# feature_importances

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np

#1. DATA
dataset = load_diabetes()
x = dataset.data 
y = dataset.target

x_pd = pd.DataFrame(x, columns=dataset['feature_names']) 
x1 = x_pd.iloc[:,0]
x2 = x_pd.iloc[:,2]
x3 = x_pd.iloc[:,4:5]
x4 = x_pd.iloc[:,8]
x = pd.concat([x1, x2, x3, x4], axis=1)
x = x.to_numpy()

x_train, x_test, y_train, y_test = \
    train_test_split(x, y, train_size=0.8, random_state=44)

#2. modeling
model = DecisionTreeRegressor(max_depth=4)

#3. Train
model.fit(x_train, y_train)

#4. Score, Predict
acc = model.score(x_test, y_test)

print("feature_importances : \n", model.feature_importances_)  
print("acc : ", acc)  

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

# feature_importances : 
#  [0.02991191 0.         0.32054901 0.         0.01831924 0.06062798
#  0.         0.         0.57059185 0.        ]
# acc :  0.31490122539834386

# feature_importances : 
#  [0.04789328 0.3230994  0.04316533 0.58584199]
# acc :  0.3002403937235434
