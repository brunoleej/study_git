# eval_set : validation0, 1을 볼 수 있음
# eval_metric : 메트릭스를 지정
# evals_result : eval_metric 에 지정했던 rmse가 줄어드는 걸 볼 수 있음
from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_boston, load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import r2_score, accuracy_score

# Data
# data, target = load_boston(return_X_y=True)

cancer = load_breast_cancer()
data = cancer.data
target = cancer.target

x_train, x_test, y_train, y_test = train_test_split(\
    data, target, test_size=0.2, shuffle=True, random_state=66)

# Modeling
model = XGBClassifier(n_estimators=10, learning_rate=0.01, n_jobs=8)

# Train
# model.fit(x_train, y_train, verbose=1, eval_metric='logloss', eval_set=[(x_train, y_train), (x_test, y_test)])
model.fit(x_train, y_train, verbose=1, eval_metric='error', eval_set=[(x_train, y_train), (x_test, y_test)])

# Evaluate
aaa = model.score(x_test, y_test)
print("aaa : ", aaa)    # acc score 와 동일

# Prediction
y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print("acc :", acc)

# logloss
# aaa :  0.9649122807017544
# acc : 0.9649122807017544

# error
# aaa :  0.9649122807017544
# acc : 0.9649122807017544

print("================")
result = model.evals_result()
print(result)   
# {'validation_0': OrderedDict([('logloss', [0.684206, 0.675503, 0.666963, 0.658583, 0.650357, 0.642281, 0.634327, 0.626517, 0.618843, 0.611307])]), 'validation_1': OrderedDict([('logloss', [0.684766, 0.676633, 0.668579, 0.660754, 0.653141, 0.645671, 0.638324, 0.631203, 0.624147, 0.617208])])}




