import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.utils.testing import all_estimators
from sklearn.datasets import load_boston
import warnings

warnings.filterwarnings('ignore')

dataset = load_boston()
data = dataset.data
target = dataset.target

x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=44)
allAlgorithms = all_estimators(type_filter='regressor')    # type_filter='classifier' : 분류형 모델 전체를 불러옴

for (name, algorithm) in allAlgorithms :   
    try :   # 에러가 없으면 아래 진행
        model = algorithm()

        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        print(name, '의 정답률 : ', r2_score(y_test, y_pred))
    except :          #에러가 발생하면
        # continue    # 정지시키지 않고 계속 진행
        print(name, "은 없는 모델") # 예외처리한 모델 이름을 출력 

# ARDRegression 의 정답률 :  0.7512651671065456
# AdaBoostRegressor 의 정답률 :  0.8524734810631045
# BaggingRegressor 의 정답률 :  0.8799781732095908
# BayesianRidge 의 정답률 :  0.7444785336818114
# CCA 의 정답률 :  0.7270542664211517
# DecisionTreeRegressor 의 정답률 :  0.8401301109480152
# DummyRegressor 의 정답률 :  -0.0007982049217318821
# ElasticNet 의 정답률 :  0.699050089875551
# ElasticNetCV 의 정답률 :  0.6902681369495264
# ExtraTreeRegressor 의 정답률 :  0.8237901614923502
# ExtraTreesRegressor 의 정답률 :  0.9013160731927451
# GammaRegressor 의 정답률 :  -0.0007982049217318821
# GaussianProcessRegressor 의 정답률 :  -5.639147690233129
# GeneralizedLinearRegressor 의 정답률 :  0.6918039703429
# GradientBoostingRegressor 의 정답률 :  0.891397319043986
# HistGradientBoostingRegressor 의 정답률 :  0.8991491407747458
# HuberRegressor 의 정답률 :  0.7043102202196916
# IsotonicRegression 은 없는 모델
# KNeighborsRegressor 의 정답률 :  0.6390759816821279
# KernelRidge 의 정답률 :  0.7744886784070626
# Lars 의 정답률 :  0.7521800808693163
# LarsCV 의 정답률 :  0.7570138649983486
# Lasso 의 정답률 :  0.6855879495660049
# LassoCV 의 정답률 :  0.71540574604873
# LassoLars 의 정답률 :  -0.0007982049217318821
# LassoLarsCV 의 정답률 :  0.7570138649983486
# LassoLarsIC 의 정답률 :  0.7540945959884459
# LinearRegression 의 정답률 :  0.752180080869314
# LinearSVR 의 정답률 :  0.33293994706857066
# MLPRegressor 의 정답률 :  0.21263083953637052
# MultiOutputRegressor 은 없는 모델
# MultiTaskElasticNet 은 없는 모델
# MultiTaskElasticNetCV 은 없는 모델
# MultiTaskLasso 은 없는 모델
# MultiTaskLassoCV 은 없는 모델
# NuSVR 의 정답률 :  0.32534704254368274
# OrthogonalMatchingPursuit 의 정답률 :  0.5661769106723642
# OrthogonalMatchingPursuitCV 의 정답률 :  0.7377665753906504
# PLSCanonical 의 정답률 :  -1.7155095545127717
# PLSRegression 의 정답률 :  0.7666940310402939
# PassiveAggressiveRegressor 의 정답률 :  -0.007122443642409548
# PoissonRegressor 의 정답률 :  0.801655416222929
# RANSACRegressor 의 정답률 :  0.5317193952559058
# RadiusNeighborsRegressor 은 없는 모델
# RandomForestRegressor 의 정답률 :  0.8936579589303041
# RegressorChain 은 없는 모델
# Ridge 의 정답률 :  0.7539303499010773
# RidgeCV 의 정답률 :  0.7530092298815411
# SGDRegressor 의 정답률 :  -3.872241875105897e+25
# SVR 의 정답률 :  0.2868662719877668
# StackingRegressor 은 없는 모델
# TheilSenRegressor 의 정답률 :  0.7907926963366596
# TransformedTargetRegressor 의 정답률 :  0.752180080869314
# TweedieRegressor 의 정답률 :  0.6918039703429
# VotingRegressor 은 없는 모델
# _SigmoidCalibration 은 없는 모델