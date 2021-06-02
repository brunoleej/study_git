# eval_set : validation0, 1을 볼 수 있음
# eval_metric : 메트릭스를 지정
# evals_result : eval_metric 에 지정했던 rmse가 줄어드는 걸 볼 수 있음
# early_stopping_rounds
# eval_graph
from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_boston, load_breast_cancer
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import r2_score, accuracy_score

# Data
# data, target = load_boston(return_X_y=True)
boston = load_boston()
data = boston.data
target = boston.target

x_train, x_test, y_train, y_test = train_test_split(\
    data, target, test_size=0.3, shuffle=True, random_state=66)

# Modeling
model = XGBRegressor(n_estimators=2000, learning_rate=0.01, n_jobs=8)

# Fitting
# model.fit(x_train, y_train, verbose=1, eval_metric='rmse', eval_set=[(x_train, y_train), (x_test, y_test)])

model.fit(x_train, y_train, verbose=1, eval_metric=['logloss','rmse'], eval_set=[(x_train, y_train), (x_test, y_test)],\
    early_stopping_rounds=30)

# Evaluate
aaa = model.score(x_test, y_test)
print("aaa : ", aaa)    # r2 score = score

# Prediction
y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
print("r2 :", r2)

# aaa :  -4.581693190871778
# r2 : -4.581693190871778

print("================")
result = model.evals_result()
print(result)   
# {'validation_0': OrderedDict([('rmse', [23.611168, 23.387598, 23.166225, 22.947048, 22.730053, 22.515182, 22.302441, 22.091829, 21.883278, 21.676794, 21.472122, 21.269697, 21.069313, 20.870634, 20.674215, 20.479692, 20.286825, 20.095963, 19.907061, 19.720037, 19.53471]), ('logloss', [-24.677284, -340.739563, -642.822021, -754.696655, -758.203796, -787.331543, 
# -790.494141, -790.50293, -790.519104, -790.543884, -791.724548, -791.724548, -791.724548, -791.724548, -791.724548, -791.724548, -791.724548, -791.724548, -791.724548, -791.724548, 
# -791.724548])]), 'validation_1': OrderedDict([('rmse', [23.777716, 23.54969, 23.323978, 23.100504, 22.87919, 22.660995, 22.444965, 22.23027, 22.018494, 21.808922, 21.599405, 21.393854, 21.193167, 20.990108, 20.793301, 20.595726, 20.398142, 20.20554, 20.014095, 19.824509, 19.6364]), ('logloss', [-26.080339, -351.378906, -657.216919, -771.809265, -773.021667, -794.158936, -794.206726, -794.244507, -794.313599, -794.4198, -799.529724, -799.529724, -799.529724, -799.529724, -799.529724, -799.529724, -799.529724, -799.529724, -799.529724, -799.529724, -799.529724])])}

import matplotlib.pyplot as plt
epochs = len(result['validation_0']['logloss'])
x_axis = range(0, epochs)

fig, ax = plt.subplots()
ax.plot(x_axis, result['validation_0']['logloss'], label='Train')
ax.plot(x_axis, result['validation_1']['logloss'], label='Test')
ax.legend()
plt.ylabel('Log Loss')
plt.title('XGBoost Log Loss')

fig, ax = plt.subplots()
ax.plot(x_axis, result['validation_0']['rmse'], label='Train')
ax.plot(x_axis, result['validation_1']['rmse'], label='Test')
ax.legend()
plt.ylabel('Rmse')
plt.title('XGBoost RMSE')

plt.show()
