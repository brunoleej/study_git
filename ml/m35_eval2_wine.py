# eval_set : validation0, 1 확인가능
# eval_metric : 메트릭스를 지정
# evals_result : eval_metric 에 지정했던 rmse가 줄어드는 것 확인가능
from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_boston, load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import r2_score, accuracy_score

# Data
# data, target = load_boston(return_X_y=True)
wine = load_wine()
data = wine.data
target = wine.target

x_train, x_test, y_train, y_test = train_test_split(\
    data, target, test_size=0.3, shuffle=True, random_state=66)

# Modeling
model = XGBClassifier(n_estimators=10, learning_rate=0.01, n_jobs=8)

# Fitting
# model.fit(x_train, y_train, verbose=1, eval_metric='mlogloss', eval_set=[(x_train, y_train), (x_test, y_test)])
model.fit(x_train, y_train, verbose=1, eval_metric='merror', eval_set=[(x_train, y_train), (x_test, y_test)])

# Evaluate
aaa = model.score(x_test, y_test)
print("aaa : ", aaa)    # r2 score 와 동일

# Prediction
y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print("acc :", acc)

# mlogloss
# aaa :  0.9722222222222222
# acc : 0.9722222222222222

# merror
# aaa :  1.0
# acc : 1.0

print("================")
result = model.evals_result()
print(result)  
# {'validation_0': OrderedDict([('mlogloss', [1.085513, 1.072629, 1.059957, 1.04749, 1.035225, 1.023157, 1.011279, 0.999589, 0.988082, 0.976753])]), 'validation_1': OrderedDict([('mlogloss', [1.085941, 1.073813, 1.061924, 1.050209, 1.038657, 1.027313, 1.016129, 1.005187, 0.994355, 0.983701])])}


