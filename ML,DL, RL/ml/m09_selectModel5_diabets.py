import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.utils.testing import all_estimators
from sklearn.datasets import load_diabetes
import warnings

warnings.filterwarnings('ignore')

dataset = load_diabetes()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=44)
allAlgorithms = all_estimators(type_filter='regressor')    # type_filter='regressor' : 회귀 모델 전체를 불러옴

for (name, algorithm) in allAlgorithms :   
    try :   # 에러가 없으면 아래 진행
        model = algorithm()

        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        print(name, '의 정답률 : ', r2_score(y_test, y_pred))
    except :          #에러가 발생하면
        # continue    # 정지시키지 않고 계속 진행
        print(name, "은 없는 모델") # 예외처리한 모델 이름을 출력 

# ARDRegression 의 정답률 :  0.5278342233068827
# AdaBoostRegressor 의 정답률 :  0.4530001690223996
# BaggingRegressor 의 정답률 :  0.3798159174502018
# BayesianRidge 의 정답률 :  0.5193410135537664
# CCA 의 정답률 :  0.4887961803882477
# DecisionTreeRegressor 의 정답률 :  -0.30735682149239385
# DummyRegressor 의 정답률 :  -0.07457975637038539
# ElasticNet 의 정답률 :  -0.06518000443720706
# ElasticNetCV 의 정답률 :  0.4294375480398558
# ExtraTreeRegressor 의 정답률 :  -0.10522540882039078
# ExtraTreesRegressor 의 정답률 :  0.43962501460544323
# GammaRegressor 의 정답률 :  -0.06869757267027454
# GaussianProcessRegressor 의 정답률 :  -16.5735788805431
# GeneralizedLinearRegressor 의 정답률 :  -0.06771406705799343
# GradientBoostingRegressor 의 정답률 :  0.3706495551620609
# HistGradientBoostingRegressor 의 정답률 :  0.3504135950167052
# HuberRegressor 의 정답률 :  0.5204968028607962
# IsotonicRegression 은 없는 모델
# KNeighborsRegressor 의 정답률 :  0.35838503635518537
# KernelRidge 의 정답률 :  -4.4187445504449405
# Lars 의 정답률 :  0.21479550446392381
# LarsCV 의 정답률 :  0.5163653521044982
# Lasso 의 정답률 :  0.33086319953362164
# LassoCV 의 정답률 :  0.5222186221789182
# LassoLars 의 정답률 :  0.3570808988866827
# LassoLarsCV 의 정답률 :  0.5214536844628463
# LassoLarsIC 의 정답률 :  0.5224736703335271
# LinearRegression 의 정답률 :  0.525204262124852
# LinearSVR 의 정답률 :  -0.8243691720358541
# MLPRegressor 의 정답률 :  -4.231720376838592
# MultiOutputRegressor 은 없는 모델
# MultiTaskElasticNet 은 없는 모델
# MultiTaskElasticNetCV 은 없는 모델
# MultiTaskLasso 은 없는 모델
# MultiTaskLassoCV 은 없는 모델
# NuSVR 의 정답률 :  0.07746639731663862
# OrthogonalMatchingPursuit 의 정답률 :  0.3337053538857254
# OrthogonalMatchingPursuitCV 의 정답률 :  0.5257611661032995
# PLSCanonical 의 정답률 :  -1.2663831979876914
# PLSRegression 의 정답률 :  0.5042012880276585
# PassiveAggressiveRegressor 의 정답률 :  0.4648828689964364
# PoissonRegressor 의 정답률 :  0.29880208432722
# RANSACRegressor 의 정답률 :  0.39393387149183223
# RadiusNeighborsRegressor 의 정답률 :  -0.07457975637038539
# RandomForestRegressor 의 정답률 :  0.37974255602065266
# RegressorChain 은 없는 모델
# Ridge 의 정답률 :  0.40179727975154833
# RidgeCV 의 정답률 :  0.5132298404989655
# SGDRegressor 의 정답률 :  0.3771856234744384
# SVR 의 정답률 :  0.008054881772852074
# StackingRegressor 은 없는 모델
# TheilSenRegressor 의 정답률 :  0.5120210165336254
# TransformedTargetRegressor 의 정답률 :  0.525204262124852
# TweedieRegressor 의 정답률 :  -0.06771406705799343
# VotingRegressor 은 없는 모델
# _SigmoidCalibration 은 없는 모델