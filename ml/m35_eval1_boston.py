# eval_set : validation0, 1을 볼 수 있다.
# eval_metric : 메트릭스를 지정한다.
# evals_result : eval_metric 에 지정했던 rmse가 줄어드는 걸 볼 수 있다

from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_boston, load_breast_cancer
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import r2_score, accuracy_score

#1. DATA
# x, y = load_boston(return_X_y=True)
datasets = load_boston()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(\
    x, y, test_size=0.2, shuffle=True, random_state=66)

#2. Modeling
model = XGBRegressor(n_estimators=10, learning_rate=0.01, n_jobs=8)

#3. Train
model.fit(x_train, y_train, verbose=1, eval_metric='rmse', eval_set=[(x_train, y_train), (x_test, y_test)])

#4. Evaluate, Predict
aaa = model.score(x_test, y_test)
print("aaa : ", aaa)    # r2 score 와 동일

y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
print("r2 :", r2)

# aaa :  -4.690504661286998
# r2 : -4.690504661286998

print("================")
result = model.evals_result()
print(result)   # evals_result : eval_metric 에 지정했던 rmse가 줄어드는 걸 볼 수 있다.
# {'validation_0': OrderedDict([('rmse', [23.611168, 23.387598, 23.166225, 22.947048, 22.730053, 22.515182, 22.302441, 22.091829, 21.883278, 21.676794])]), 'validation_1': OrderedDict([('rmse', [23.777716, 23.54969, 23.323978, 23.100504, 22.87919, 22.660995, 22.444965, 22.23027, 22.018494, 21.808922])])}
