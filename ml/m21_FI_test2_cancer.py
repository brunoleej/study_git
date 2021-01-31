# feature_importances
# 중요도가 낮은 컬럼 제거한 후 실행 >> 제거하기 전이랑 결과 유사하다

from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

#1. DATA
dataset = load_breast_cancer()
x = dataset.data 
y = dataset.target

x_pd = pd.DataFrame(x, columns=dataset['feature_names']) 
x1 = x_pd.iloc[:,21:25]
x2 = x_pd.iloc[:,27]
x = pd.concat([x1, x2], axis=1)
x = x.to_numpy()

# print(x.shape)
# print(y.shape)

x_train, x_test, y_train, y_test = \
    train_test_split(x, y, train_size=0.8, random_state=44)

#2. modeling
model = DecisionTreeClassifier(max_depth=4)

#3. Train
model.fit(x_train, y_train)

#4. Score, Predict
acc = model.score(x_test, y_test)

print("feature importances : \n", model.feature_importances_)  
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

# feature importances : 
# [0.         0.         0.         0.         0.         0.
#  0.         0.         0.         0.         0.         0.
#  0.         0.         0.         0.         0.         0.
#  0.         0.         0.         0.05612587 0.78678449 0.01008994
#  0.02293065 0.         0.         0.12406904 0.         0.        ]
# acc :  0.9385964912280702

# 중요도 0인 컬럼 삭제한 후 >> 삭제하거나 그대로나 acc 유사하다
# feature importances :
#  [0.05612587 0.78000877 0.01686566 0.00995429 0.1370454 ]
# acc :  0.9385964912280702