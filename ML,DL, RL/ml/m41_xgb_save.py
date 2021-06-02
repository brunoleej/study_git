# xgboost model save
# 3. xgb save_model

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
model = XGBRegressor(n_estimators=1000, learning_rate=0.01, n_jobs=8)

#3. Train
# model.fit(x_train, y_train, verbose=1, eval_metric='rmse', eval_set=[(x_train, y_train), (x_test, y_test)])

# eval_metirc에 여러개 메트릭스를 넣을 수 있다. 리스트 사용
model.fit(x_train, y_train, verbose=1, eval_metric=['rmse'], eval_set=[(x_train, y_train), (x_test, y_test)],\
    early_stopping_rounds=10)
#  early_stopping_rounds=10번 돌리고 끝낸다.
#  params에 제공된 eval_metric 매개 변수 에 두 개 이상의 측정 항목이있는 경우 마지막 측정 항목이 조기 중지에 사용

#4. Evaluate, Predict
aaa = model.score(x_test, y_test)
print("model.score : ", aaa)    # r2 score 와 동일

y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
print("r2 :", r2)

# aaa :  -4.690504661286998
# r2 : -4.690504661286998

print("================")
result = model.evals_result()
# print(result)   # evals_result : eval_metric 에 지정했던 rmse가 줄어드는 걸 볼 수 있다.
# {'validation_0': OrderedDict([('rmse', [23.611168, 23.387598, 23.166225, 22.947048, 22.730053, 22.515182, 22.302441, 22.091829, 21.883278, 21.676794])]), 'validation_1': OrderedDict([('rmse', [23.777716, 23.54969, 23.323978, 23.100504, 22.87919, 22.660995, 22.444965, 22.23027, 22.018494, 21.808922])])}

# [1]
import pickle
# model & weight save
# pickle.dump(model, open('../data/xgb_save/m39.pickle.data', 'wb')) # wb : write
# print("== save complete ==")

# [2]
import joblib
# joblib.dump(model, '../data/xgb_save/m39.joblib.data')

# [3]
# model.save_model("../data/xgb_save/m39.xgb.model")

'''
print("========pickle load========")
# model laod
model2 = pickle.load(open('../data/xgb_save/m39.pickle.data', 'rb'))  # rb : read
print("== load complete ==")
r22 = model2.score(x_test, y_test)
print("r22 : ", r22)


print("========joblib load========")
# model laod
model2 = joblib.load('../data/xgb_save/m39.joblib.data')  # rb : read
print("== load complete ==")
r22 = model2.score(x_test, y_test)
print("r22 : ", r22)
# r22 :  0.93302293398985
'''

print("========xgb save model load========")
# model laod
model2 = XGBRegressor() # 사용할 모델을 다시 한 번 지정해준다.
model2.load_model('../data/xgb_save/m39.xgb.model')
r22 = model2.score(x_test, y_test)
print("r22 : ", r22)
# r22 :  0.93302293398985