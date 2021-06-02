import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators
from sklearn.datasets import load_wine
import warnings

warnings.filterwarnings('ignore')   # 경고 메세지를 무시함

wine = load_wine()
data = wine.data
target = wine.target

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
#  [0.93103448 0.82758621 0.92857143 0.89285714 1.        ]
# BaggingClassifier 의 정답률 : 
#  [1.         0.86206897 0.92857143 0.92857143 0.96428571]
# BernoulliNB 의 정답률 : 
#  [0.31034483 0.37931034 0.39285714 0.35714286 0.53571429]
# CalibratedClassifierCV 의 정답률 : 
#  [0.89655172 1.         0.96428571 0.92857143 0.82142857]
# CategoricalNB 은 없는 모델
# CheckingClassifier 의 정답률 : 
#  [0. 0. 0. 0. 0.]
# ClassifierChain 은 없는 모델
# ComplementNB 의 정답률 : 
#  [0.65517241 0.65517241 0.75       0.57142857 0.75      ]
# DecisionTreeClassifier 의 정답률 : 
#  [0.82758621 0.86206897 0.89285714 0.85714286 0.89285714]
# DummyClassifier 의 정답률 : 
#  [0.13793103 0.31034483 0.39285714 0.21428571 0.25      ]
# ExtraTreeClassifier 의 정답률 : 
#  [0.93103448 0.93103448 0.82142857 0.89285714 0.85714286]
# ExtraTreesClassifier 의 정답률 : 
#  [1. 1. 1. 1. 1.]
# GaussianNB 의 정답률 : 
#  [1.         0.96551724 1.         0.96428571 1.        ]
# GaussianProcessClassifier 의 정답률 : 
#  [0.37931034 0.48275862 0.42857143 0.5        0.42857143]
# GradientBoostingClassifier 의 정답률 : 
#  [0.86206897 0.89655172 0.96428571 0.96428571 0.92857143]
# HistGradientBoostingClassifier 의 정답률 : 
#  [1.         1.         0.92857143 0.96428571 1.        ]
# KNeighborsClassifier 의 정답률 : 
#  [0.65517241 0.5862069  0.53571429 0.75       0.85714286]
# LabelPropagation 의 정답률 : 
#  [0.51724138 0.34482759 0.39285714 0.35714286 0.39285714]
# LabelSpreading 의 정답률 : 
#  [0.48275862 0.27586207 0.42857143 0.39285714 0.42857143]
# LinearDiscriminantAnalysis 의 정답률 : 
#  [1.         0.96551724 1.         0.96428571 0.96428571]
# LinearSVC 의 정답률 : 
#  [0.75862069 0.86206897 0.60714286 0.67857143 0.89285714]
# LogisticRegression 의 정답률 : 
#  [0.89655172 1.         0.96428571 0.96428571 0.89285714]
# LogisticRegressionCV 의 정답률 : 
#  [0.96551724 1.         0.96428571 0.89285714 0.92857143]
# MLPClassifier 의 정답률 : 
#  [0.17241379 0.5862069  0.92857143 0.57142857 0.67857143]
# MultiOutputClassifier 은 없는 모델
# MultinomialNB 의 정답률 : 
#  [0.93103448 0.79310345 0.71428571 0.89285714 0.89285714]
# NearestCentroid 의 정답률 : 
#  [0.86206897 0.68965517 0.67857143 0.82142857 0.71428571]
# NuSVC 의 정답률 : 
#  [0.89655172 0.75862069 0.89285714 0.89285714 0.89285714]
# OneVsOneClassifier 은 없는 모델
# OneVsRestClassifier 은 없는 모델
# OutputCodeClassifier 은 없는 모델
# PassiveAggressiveClassifier 의 정답률 : 
#  [0.48275862 0.51724138 0.64285714 0.75       0.32142857]
# Perceptron 의 정답률 : 
#  [0.55172414 0.37931034 0.5        0.71428571 0.53571429]
# QuadraticDiscriminantAnalysis 의 정답률 : 
#  [1.         0.96551724 0.92857143 1.         1.        ]
# RadiusNeighborsClassifier 은 없는 모델
# RandomForestClassifier 의 정답률 : 
#  [1.         1.         0.96428571 1.         0.96428571]
# RidgeClassifier 의 정답률 : 
#  [1.         1.         1.         1.         0.96428571]
# RidgeClassifierCV 의 정답률 : 
#  [1.         0.96551724 0.96428571 1.         1.        ]
# SGDClassifier 의 정답률 : 
#  [0.55172414 0.65517241 0.53571429 0.67857143 0.32142857]
# SVC 의 정답률 : 
#  [0.82758621 0.65517241 0.82142857 0.67857143 0.60714286]
# StackingClassifier 은 없는 모델
# VotingClassifier 은 없는 모델