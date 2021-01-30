import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score, r2_score
from sklearn.utils.testing import all_estimators
from sklearn.datasets import load_diabetes
import warnings

warnings.filterwarnings('ignore')   # 오류메시지를 무시함

diabetes = load_diabetes()
data = diabetes.data
target = diabetes.target

x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=44)
kfold = KFold(n_splits=5, shuffle=True)

allAlgorithms = all_estimators(type_filter='regressor')    # type_filter='regressor' : 회귀형 모델 전체를 불러옴

for (name, algorithm) in allAlgorithms :    
    try :   
        model = algorithm()
        score = cross_val_score(model, x_train, y_train, cv=kfold)
        # model.fit(x_train, y_train)
        # y_pred = model.predict(x_test)
        print(name, '의 정답률 : \n', score)
    except :          
        # continue   
        print(name, "은 없는 모델") 