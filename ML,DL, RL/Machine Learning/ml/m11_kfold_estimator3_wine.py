import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators
from sklearn.datasets import load_breast_cancer
import warnings

warnings.filterwarnings('ignore')   # 경고 메세지를 무시함

cancer = load_breast_cancer()
data = cancer.data
target = cancer.target

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
#  [0.95604396 0.96703297 0.9010989  0.95604396 0.95604396]
# BaggingClassifier 의 정답률 : 
#  [0.97802198 0.9010989  0.91208791 0.98901099 0.95604396]
# BernoulliNB 의 정답률 : 
#  [0.69230769 0.59340659 0.62637363 0.62637363 0.56043956]
# CalibratedClassifierCV 의 정답률 : 
#  [0.91208791 0.86813187 0.94505495 0.92307692 0.89010989]
# CategoricalNB 은 없는 모델
# CheckingClassifier 의 정답률 : 
#  [0. 0. 0. 0. 0.]
# ClassifierChain 은 없는 모델
# ComplementNB 의 정답률 : 
#  [0.87912088 0.95604396 0.84615385 0.87912088 0.82417582]
# DecisionTreeClassifier 의 정답률 : 
#  [0.92307692 0.91208791 0.92307692 0.9010989  0.91208791]
# DummyClassifier 의 정답률 : 
#  [0.48351648 0.49450549 0.53846154 0.51648352 0.47252747]
# ExtraTreeClassifier 의 정답률 : 
#  [0.92307692 0.91208791 0.87912088 0.91208791 0.9010989 ]
# ExtraTreesClassifier 의 정답률 : 
#  [0.97802198 0.98901099 0.96703297 0.93406593 0.97802198]
# GaussianNB 의 정답률 : 
#  [0.9010989  0.92307692 0.95604396 0.9010989  0.95604396]
# GaussianProcessClassifier 의 정답률 : 
#  [0.9010989  0.91208791 0.92307692 0.92307692 0.91208791]
# GradientBoostingClassifier 의 정답률 : 
#  [0.97802198 0.92307692 0.92307692 0.96703297 0.96703297]
# HistGradientBoostingClassifier 의 정답률 : 
#  [0.98901099 0.96703297 0.92307692 0.96703297 0.97802198]
# KNeighborsClassifier 의 정답률 : 
#  [0.92307692 0.9010989  0.92307692 0.95604396 0.89010989]
# LabelPropagation 의 정답률 : 
#  [0.31868132 0.43956044 0.35164835 0.3956044  0.41758242]
# LabelSpreading 의 정답률 : 
#  [0.3956044  0.47252747 0.3956044  0.31868132 0.38461538]
# LinearDiscriminantAnalysis 의 정답률 : 
#  [0.97802198 0.94505495 0.91208791 0.96703297 0.93406593]
# LinearSVC 의 정답률 : 
#  [0.94505495 0.89010989 0.93406593 0.95604396 0.86813187]
# LogisticRegression 의 정답률 : 
#  [0.92307692 0.91208791 0.95604396 0.9010989  0.94505495]
# LogisticRegressionCV 의 정답률 : 
#  [0.9010989  0.98901099 0.95604396 0.96703297 0.95604396]
# MLPClassifier 의 정답률 : 
#  [0.92307692 0.85714286 0.95604396 0.96703297 0.91208791]
# MultiOutputClassifier 은 없는 모델
# MultinomialNB 의 정답률 : 
#  [0.9010989  0.9010989  0.86813187 0.9010989  0.84615385]
# NearestCentroid 의 정답률 : 
#  [0.86813187 0.91208791 0.91208791 0.85714286 0.86813187]
# NuSVC 의 정답률 : 
#  [0.86813187 0.89010989 0.86813187 0.83516484 0.9010989 ]
# OneVsOneClassifier 은 없는 모델
# OneVsRestClassifier 은 없는 모델
# OutputCodeClassifier 은 없는 모델
# PassiveAggressiveClassifier 의 정답률 : 
#  [0.82417582 0.9010989  0.78021978 0.92307692 0.94505495]
# Perceptron 의 정답률 : 
#  [0.83516484 0.89010989 0.91208791 0.76923077 0.82417582]
# QuadraticDiscriminantAnalysis 의 정답률 : 
#  [0.96703297 0.97802198 0.95604396 0.95604396 0.9010989 ]
# RadiusNeighborsClassifier 은 없는 모델
# RandomForestClassifier 의 정답률 : 
#  [0.94505495 0.97802198 0.95604396 0.95604396 1.        ]
# RidgeClassifier 의 정답률 : 
#  [0.94505495 0.92307692 0.94505495 0.95604396 0.95604396]
# RidgeClassifierCV 의 정답률 : 
#  [0.93406593 0.95604396 0.98901099 0.94505495 0.96703297]
# SGDClassifier 의 정답률 : 
#  [0.75824176 0.78021978 0.82417582 0.73626374 0.78021978]
# SVC 의 정답률 : 
#  [0.94505495 0.86813187 0.86813187 0.89010989 0.95604396]
# StackingClassifier 은 없는 모델
# VotingClassifier 은 없는 모델