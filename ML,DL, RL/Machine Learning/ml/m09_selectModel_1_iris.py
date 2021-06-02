import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators
from sklearn.datasets import load_iris
import warnings

warnings.filterwarnings('ignore')

dataset = load_iris()
data = dataset.data
target = dataset.target

x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=44)
allAlgorithms = all_estimators(type_filter='classifier')    # type_filter='classifier' : 분류형 모델 전체를 불러옴

for (name, algorithm) in allAlgorithms :    
    try :   # 에러가 없으면 아래 진행
        model = algorithm()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        print(name, '의 정답률 : ', accuracy_score(y_test, y_pred))
    except :          
        # continue    # 정지시키지 않고 계속 진행
        print(name, "은 없는 모델") # 예외처리한 모델 이름을 출력 

# AdaBoostClassifier 의 정답률 :  0.9666666666666667
# BaggingClassifier 의 정답률 :  0.9666666666666667
# BernoulliNB 의 정답률 :  0.3
# CalibratedClassifierCV 의 정답률 :  0.9333333333333333
# CategoricalNB 의 정답률 :  0.9
# CheckingClassifier 의 정답률 :  0.3
# ClassifierChain 은 없는 모델
# ComplementNB 의 정답률 :  0.7
# DecisionTreeClassifier 의 정답률 :  0.9666666666666667
# DummyClassifier 의 정답률 :  0.36666666666666664
# ExtraTreeClassifier 의 정답률 :  0.9666666666666667
# ExtraTreesClassifier 의 정답률 :  0.9666666666666667
# GaussianNB 의 정답률 :  0.9333333333333333
# GaussianProcessClassifier 의 정답률 :  0.9666666666666667
# GradientBoostingClassifier 의 정답률 :  0.9666666666666667
# HistGradientBoostingClassifier 의 정답률 :  0.9666666666666667
# KNeighborsClassifier 의 정답률 :  0.9666666666666667
# LabelPropagation 의 정답률 :  0.9666666666666667
# LabelSpreading 의 정답률 :  0.9666666666666667
# LinearDiscriminantAnalysis 의 정답률 :  1.0
# LinearSVC 의 정답률 :  0.9666666666666667
# LogisticRegression 의 정답률 :  0.9666666666666667
# LogisticRegressionCV 의 정답률 :  0.9666666666666667
# MLPClassifier 의 정답률 :  1.0
# MultiOutputClassifier 은 없는 모델
# MultinomialNB 의 정답률 :  0.8666666666666667
# NearestCentroid 의 정답률 :  0.9
# NuSVC 의 정답률 :  0.9666666666666667
# OneVsOneClassifier 은 없는 모델
# OneVsRestClassifier 은 없는 모델
# OutputCodeClassifier 은 없는 모델
# PassiveAggressiveClassifier 의 정답률 :  0.9
# Perceptron 의 정답률 :  0.7333333333333333
# QuadraticDiscriminantAnalysis 의 정답률 :  1.0
# RadiusNeighborsClassifier 의 정답률 :  0.9333333333333333
# RandomForestClassifier 의 정답률 :  0.9666666666666667
# RidgeClassifier 의 정답률 :  0.8333333333333334
# RidgeClassifierCV 의 정답률 :  0.8333333333333334
# SGDClassifier 의 정답률 :  1.0
# SVC 의 정답률 :  0.9666666666666667
# StackingClassifier 은 없는 모델
# VotingClassifier 은 없는 모델