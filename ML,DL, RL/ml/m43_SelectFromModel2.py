# [1] 상단모델에 그리드서치 혹은 랜덤서치로 튜닝한 모델 구성
# > 최적의 r2, feature_importances 구할 것

# [2] 위 값으로 selectfrommodel을 구해서 최적의 피처 개수를 구할 것

# [3] 위 피처 개수로 데이터(피처)를 수정(삭제)해서 그리드서치, 랜덤서리 적용하여 최적의 r2 구할 것
 
from xgboost import XGBClassifier, XGBRegressor, plot_importance
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV, GridSearchCV
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import r2_score, accuracy_score

print("[1] Basic ==============================")

#1. DATA
x, y = load_boston(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

parameters = [
    {"n_estimators":[90, 100, 110, 120], "learning_rate":[0.1, 0.05, 0.01, 0.005, 0.001], 
    "max_depth":[3,5,7],"colsample_bytree":[0.6, 0.9, 1], "colsample_bylevel":[0.6, 0.7, 0.9]}
]

#2. Modeling
model = RandomizedSearchCV(XGBRegressor(n_jobs=8, use_label_encoder=False), parameters, cv=kf)

#3. Train
model.fit(x_train, y_train, eval_metric='logloss')

#4. Score, Preidct
score = model.score(x_test, y_test)
print("model.score : ", score)

y_pred = model.predict(x_test)
r2 = r2_score(y_pred, y_test)
print("r2_score : ", r2)

# model.score :  0.9447618613737424
# r2_score :  0.9408009654559968

print("[2] SelectFromModel ==============================")

bs = model.best_estimator_
best = bs.fit(x_train, y_train)

thresholds = np.sort(model.best_estimator_.feature_importances_)  
# print("thresholds ", thresholds)
# [0.00353304 0.00935275 0.01063521 0.01271748 0.01771014 0.02028915
#  0.03412732 0.03611462 0.04129329 0.08856925 0.1971374  0.22804722
#  0.30047306]

best_tmp = [0,0]
best_r2 = 0
best_feature = []

df = pd.DataFrame(x)

for thresh in thresholds :
    # selection = SelectFromModel(model, threshold=thresh, prefit=True)
    selection = SelectFromModel(best, threshold=thresh, prefit=True)

    feature_idx = selection.get_support()
    # print(feature_idx)

    feature_name = df.columns[feature_idx]
    # print(feature_name)
    select_x_train = selection.transform(x_train)
    # print(select_x_train.shape)

    selection_model = XGBRegressor(n_jobs=8)
    selection_model.fit(select_x_train, y_train)

    select_x_test = selection.transform(x_test)

    y_predict = selection_model.predict(select_x_test)

    score = r2_score(y_test, y_predict)

    # print("Thresh=%.3f, n=%d, R2: %.2f%%" % (thresh, select_x_train.shape[1], score*100))

    if best_r2 < score :
        best_tmp[0] = thresh
        best_tmp[1] = select_x_train.shape[1]
        best_r2 = score
        best_feature = feature_name


print("Thresh=%.3f, n=%d, R2: %.2f%%" % (best_tmp[0], best_tmp[1], best_r2*100))

# best_feature = best_feature.to_numpy()
# print(best_feature)

print("[3] After SelectFromModel==============================")

'''
# 가장 결과가 좋은 컬럼만 남기고 다시 훈련시키는 방법
X = x[:,best_feature]
print(X.shape)
x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=True, random_state=66)

#2. Modeling
model = RandomizedSearchCV(XGBRegressor(n_jobs=8, use_label_encoder=False), parameters, cv=kf)

#3. Train
model.fit(x_train, y_train, eval_metric='logloss')

#4. Score, Preidct
score = model.score(x_test, y_test)
print("model.score : ", score)

y_pred = model.predict(x_test)
r2 = r2_score(y_pred, y_test)
print("r2_score : ", r2)
'''

selection_f = SelectFromModel(best, threshold=best_tmp[0], prefit=True)

select_x_train_f = selection_f.transform(x_train)
print(select_x_train_f.shape)

selection_model_f = RandomizedSearchCV(XGBRegressor(n_jobs=8), parameters, cv=kf)
selection_model_f.fit(select_x_train_f, y_train)

select_x_test_f = selection_f.transform(x_test)
y_predict_f = selection_model_f.predict(select_x_test_f)
score_f = r2_score(y_test, y_predict_f)
print("Thresh=%.3f, n=%d, R2: %.2f%%" % (best_tmp[0], select_x_train_f.shape[1], score_f*100))

# [1] Basic ==============================
# model.score :  0.9359672357370028
# r2_score :  0.9314630643960781
# [2] SelectFromModel ==============================
# Thresh=0.023, n=7, R2: 92.86%
# [3] After SelectFromModel==============================
# (404, 7)
# Thresh=0.023, n=7, R2: 93.39%