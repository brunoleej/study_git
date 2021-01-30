import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators
from sklearn.datasets import load_iris
import warnings

warnings.filterwarnings('ignore')   # 경고 메세지를 무시함

iris = load_iris()
data = iris.data
target = iris.target

x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=44)
kfold = KFold(n_splits=5, shuffle=True)

allAlgorithms = all_estimators(type_filter='classifier')    # type_filter='classifier' : 분류형 모델 전체를 불러옴

for (name, algorithm) in allAlgorithms :   
    try :   
        model = algorithm()
        scores = cross_val_score(model, x_train, y_train, cv=kfold)  
        # model.fit(x_train, y_train)
        # y_pred = model.predict(x_test)
        print(name, '의 정답률 : \n', scores) 
    except :         
        # continue    
        print(name, "은 없는 모델") 

# AdaBoostClassifier 의 정답률 : 
#  [0.95833333 0.91666667 0.95833333 1.         0.875     ]
# BaggingClassifier 의 정답률 : 
#  [0.875      0.91666667 1.         0.91666667 0.95833333]
# BernoulliNB 의 정답률 : 
#  [0.25       0.33333333 0.20833333 0.20833333 0.20833333]
# CalibratedClassifierCV 의 정답률 : 
#  [0.91666667 0.875      0.875      0.91666667 0.91666667]
# CategoricalNB 의 정답률 : 
#  [0.95833333 0.95833333 0.91666667 0.95833333 0.875     ]
# CheckingClassifier 의 정답률 : 
#  [0. 0. 0. 0. 0.]
# ClassifierChain 은 없는 모델
# ComplementNB 의 정답률 : 
#  [0.75       0.54166667 0.75       0.625      0.625     ]
# DecisionTreeClassifier 의 정답률 : 
#  [0.95833333 0.91666667 0.91666667 0.95833333 0.95833333]
# DummyClassifier 의 정답률 : 
#  [0.29166667 0.33333333 0.45833333 0.375      0.41666667]
# ExtraTreeClassifier 의 정답률 : 
#  [0.875      0.95833333 1.         0.875      0.875     ]
# ExtraTreesClassifier 의 정답률 : 
#  [0.95833333 1.         0.91666667 0.91666667 0.95833333]
# GaussianNB 의 정답률 : 
#  [0.91666667 0.95833333 1.         1.         0.95833333]
# GaussianProcessClassifier 의 정답률 : 
#  [0.91666667 1.         0.95833333 0.91666667 0.95833333]
# GradientBoostingClassifier 의 정답률 : 
#  [0.95833333 1.         0.91666667 0.91666667 0.91666667]
# HistGradientBoostingClassifier 의 정답률 : 
#  [0.95833333 1.         0.91666667 0.95833333 0.83333333]
# KNeighborsClassifier 의 정답률 : 
#  [0.95833333 1.         0.91666667 1.         0.91666667]
# LabelPropagation 의 정답률 : 
#  [0.91666667 0.91666667 1.         1.         0.95833333]
# LabelSpreading 의 정답률 : 
#  [0.91666667 0.91666667 1.         1.         1.        ]
# LinearDiscriminantAnalysis 의 정답률 : 
#  [0.95833333 0.91666667 1.         1.         0.91666667]
# LinearSVC 의 정답률 : 
#  [0.875      1.         0.91666667 1.         0.95833333]
# LogisticRegression 의 정답률 : 
#  [0.875      0.95833333 1.         1.         0.91666667]
# LogisticRegressionCV 의 정답률 : 
#  [0.95833333 1.         0.95833333 0.95833333 0.95833333]
# MLPClassifier 의 정답률 : 
#  [0.95833333 1.         1.         0.95833333 0.91666667]
# MultiOutputClassifier 은 없는 모델
# MultinomialNB 의 정답률 : 
#  [0.70833333 0.625      0.54166667 0.41666667 0.95833333]
# NearestCentroid 의 정답률 : 
#  [0.95833333 0.83333333 1.         1.         0.875     ]
# NuSVC 의 정답률 : 
#  [1.         1.         0.875      0.95833333 0.95833333]
# OneVsOneClassifier 은 없는 모델
# OneVsRestClassifier 은 없는 모델
# OutputCodeClassifier 은 없는 모델
# PassiveAggressiveClassifier 의 정답률 : 
#  [0.875      0.75       0.66666667 0.79166667 0.45833333]
# Perceptron 의 정답률 : 
#  [0.33333333 0.875      0.75       0.5        0.70833333]
# QuadraticDiscriminantAnalysis 의 정답률 : 
#  [1.         0.95833333 1.         1.         0.91666667]
# RadiusNeighborsClassifier 의 정답률 : 
#  [0.95833333 0.95833333 0.95833333 1.         0.875     ]
# RandomForestClassifier 의 정답률 : 
#  [0.95833333 0.95833333 0.95833333 0.91666667 0.91666667]
# RidgeClassifier 의 정답률 : 
#  [0.91666667 0.875      0.83333333 0.83333333 0.91666667]
# RidgeClassifierCV 의 정답률 : 
#  [0.875      0.83333333 0.91666667 0.75       0.83333333]
# SGDClassifier 의 정답률 : 
#  [0.66666667 0.875      0.875      0.75       0.45833333]
# SVC 의 정답률 : 
#  [0.95833333 1.         0.91666667 0.875      0.91666667]
# StackingClassifier 은 없는 모델
# VotingClassifier 은 없는 모델