import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score, r2_score
from sklearn.utils.testing import all_estimators
from sklearn.datasets import load_boston
import warnings

warnings.filterwarnings('ignore')   # 오류메시지를 무시함

boston = load_boston()
data = dataset.data
target = dataset.target

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

# ARDRegression 의 정답률 : 
#  [0.64449266 0.70218796 0.70837816 0.64414081 0.69503173]
# AdaBoostRegressor 의 정답률 : 
#  [0.80238996 0.85637482 0.84666883 0.79875392 0.63278022]
# BaggingRegressor 의 정답률 : 
#  [0.90365273 0.83630025 0.85944391 0.74539062 0.8630094 ]
# BayesianRidge 의 정답률 : 
#  [0.47336754 0.78555061 0.68514543 0.63456389 0.78790455]
# CCA 의 정답률 : 
#  [0.63028155 0.68547352 0.68738745 0.71381261 0.60288337]
# DecisionTreeRegressor 의 정답률 : 
#  [0.40836241 0.8360773  0.60162279 0.84014123 0.74023643]
# DummyRegressor 의 정답률 : 
#  [-6.84494317e-04 -1.13870780e-03 -3.34858623e-03 -2.33275412e-03
#  -5.03620488e-05]
# ElasticNet 의 정답률 : 
#  [0.6846618  0.63554192 0.7322376  0.51435092 0.63882478]
# ElasticNetCV 의 정답률 : 
#  [0.68942246 0.66472795 0.58032898 0.66390457 0.58898108]
# ExtraTreeRegressor 의 정답률 : 
#  [0.65830017 0.60700137 0.617891   0.51642237 0.76286916]
# ExtraTreesRegressor 의 정답률 : 
#  [0.88324998 0.90881003 0.87739061 0.81850884 0.90518312]
# GammaRegressor 의 정답률 : 
#  [-0.05468407 -0.00030421 -0.01670138 -0.00206381 -0.01625338]
# GaussianProcessRegressor 의 정답률 : 
#  [-6.29338982 -4.88558414 -5.54110444 -7.81891908 -6.57257409]
# GeneralizedLinearRegressor 의 정답률 : 
#  [0.68275195 0.61066967 0.64904975 0.67868704 0.62243683]
# GradientBoostingRegressor 의 정답률 : 
#  [0.76744652 0.89730615 0.91289225 0.87994936 0.89850128]
# HistGradientBoostingRegressor 의 정답률 : 
#  [0.84639881 0.88634359 0.83829545 0.88141416 0.76641206]
# HuberRegressor 의 정답률 : 
#  [0.65409082 0.62613498 0.50104003 0.63958237 0.50994728]
# IsotonicRegression 의 정답률 : 
#  [nan nan nan nan nan]
# KNeighborsRegressor 의 정답률 : 
#  [0.44418757 0.49683871 0.55039105 0.33677211 0.45566564]
# KernelRidge 의 정답률 : 
#  [0.66687177 0.70184083 0.64043657 0.67410647 0.46682398]
# Lars 의 정답률 : 
#  [0.62750069 0.75829325 0.76023643 0.65577419 0.68429551]
# LarsCV 의 정답률 : 
#  [0.69133532 0.73919284 0.69477831 0.75804034 0.52382814]
# Lasso 의 정답률 : 
#  [0.69502414 0.68698939 0.63553014 0.60041288 0.59041057]
# LassoCV 의 정답률 : 
#  [0.73460609 0.63829476 0.61986107 0.59204144 0.67970232]
# LassoLars 의 정답률 : 
#  [-0.00046915 -0.03473862 -0.01010563 -0.0393931  -0.0104372 ]
# LassoLarsCV 의 정답률 : 
#  [0.61164699 0.69527279 0.79642783 0.75849452 0.67461927]
# LassoLarsIC 의 정답률 : 
#  [0.71326394 0.75879698 0.71744704 0.72815602 0.46380162]
# LinearRegression 의 정답률 : 
#  [0.81173168 0.68520342 0.68735654 0.5479812  0.69257136]
# LinearSVR 의 정답률 : 
#  [ 0.6474584   0.44841046 -0.17554871  0.40532503  0.61030974]
# MLPRegressor 의 정답률 : 
#  [0.6067325  0.55940155 0.40121007 0.60937339 0.46601509]
# MultiOutputRegressor 은 없는 모델
# MultiTaskElasticNet 의 정답률 : 
#  [nan nan nan nan nan]
# MultiTaskElasticNetCV 의 정답률 : 
#  [nan nan nan nan nan]
# MultiTaskLasso 의 정답률 : 
#  [nan nan nan nan nan]
# MultiTaskLassoCV 의 정답률 : 
#  [nan nan nan nan nan]
# NuSVR 의 정답률 : 
#  [0.0751109  0.19053217 0.20428952 0.18462456 0.30718912]
# OrthogonalMatchingPursuit 의 정답률 : 
#  [0.47789831 0.55893158 0.57947255 0.5206647  0.45814161]
# OrthogonalMatchingPursuitCV 의 정답률 : 
#  [0.68585617 0.74340242 0.48093825 0.67169663 0.6820988 ]
# PLSCanonical 의 정답률 : 
#  [-1.66204176 -2.31107976 -2.79076346 -2.39589253 -1.90774948]
# PLSRegression 의 정답률 : 
#  [0.63361109 0.71854585 0.65649401 0.64745135 0.69517936]
# PassiveAggressiveRegressor 의 정답률 : 
#  [-0.34127282  0.25176394  0.12016852 -0.22883129 -0.15739997]
# PoissonRegressor 의 정답률 : 
#  [0.79037019 0.71256573 0.82208784 0.74234385 0.65550236]
# RANSACRegressor 의 정답률 : 
#  [0.47745439 0.06017156 0.36980013 0.16283163 0.59513223]
# RadiusNeighborsRegressor 은 없는 모델
# RandomForestRegressor 의 정답률 : 
#  [0.79781072 0.86703185 0.88552014 0.92159651 0.86480535]
# RegressorChain 은 없는 모델
# Ridge 의 정답률 : 
#  [0.75153307 0.71620874 0.66604709 0.71833924 0.63895449]
# RidgeCV 의 정답률 : 
#  [0.64799003 0.62980882 0.7193604  0.72784305 0.77845713]
# SGDRegressor 의 정답률 : 
#  [-1.64124469e+26 -1.23436750e+26 -3.70795507e+26 -3.48765194e+26
#  -2.48823085e+26]
# SVR 의 정답률 : 
#  [-0.01918179  0.27621872  0.27714314  0.09602512  0.21036107]
# StackingRegressor 은 없는 모델
# TheilSenRegressor 의 정답률 : 
#  [0.70316909 0.61156378 0.78526932 0.61946259 0.65955128]
# TransformedTargetRegressor 의 정답률 : 
#  [0.69611548 0.73429519 0.72693692 0.63073973 0.6676868 ]
# TweedieRegressor 의 정답률 : 
#  [0.66397237 0.58733894 0.71538005 0.64156004 0.6107551 ]
# VotingRegressor 은 없는 모델
# _SigmoidCalibration 의 정답률 : 
#  [nan nan nan nan nan]