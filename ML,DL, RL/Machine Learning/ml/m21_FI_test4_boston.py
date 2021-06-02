# feature_importances
# 중요도 낮은 컬럼 제거 후 실행 >> 없애기 전이랑 비슷함
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# Data
boston = load_boston()
data = boston.data 
target = boston.target

data_df = pd.DataFrame(data, columns=boston['feature_names']) 
data1 = data_df.iloc[:,4:8]
data2 = data_df.iloc[:,10:13]
data = pd.concat([data1, data2], axis=1)
data = data.to_numpy()

x_train, x_test, y_train, y_test = \
    train_test_split(data, target, test_size = 0.3, random_state=44)

#2. modeling
model = DecisionTreeRegressor(max_depth=4)

#3. Train
model.fit(x_train, y_train)

#4. Score, Predict
acc = model.score(x_test, y_test)

print("feature_importances : \n", model.feature_importances_)  
print("acc : ", acc)  

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
#  [0.04653374 0.6014209  0.00332314 0.07961838 0.02539567 0.01820353
#  0.22550464]
# acc :  0.844136178010803