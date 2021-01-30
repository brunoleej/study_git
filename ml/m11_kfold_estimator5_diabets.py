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

# ARDRegression 의 정답률 : 
#  [0.51861811 0.25719974 0.5260789  0.54562921 0.4283656 ]
# AdaBoostRegressor 의 정답률 : 
#  [0.55726828 0.27856693 0.54868891 0.4158762  0.31850875]
# BaggingRegressor 의 정답률 : 
#  [0.31897515 0.45245333 0.21593177 0.43426189 0.20786599]
# BayesianRidge 의 정답률 : 
#  [0.44571328 0.4386771  0.26450771 0.44587411 0.61084084]
# CCA 의 정답률 : 
#  [ 0.44818544  0.53838068  0.40914633 -0.04098308  0.41505005]
# DecisionTreeRegressor 의 정답률 : 
#  [ 0.24302134 -0.13613934  0.14499254 -0.00303152 -0.06394567]
# DummyRegressor 의 정답률 : 
#  [-4.02544310e-02 -6.83579326e-03 -2.05948388e-05 -3.81984312e-02
#  -7.26487821e-03]
# ElasticNet 의 정답률 : 
#  [ 0.00797077 -0.08084508 -0.00161567 -0.00580915 -0.00160197]
# ElasticNetCV 의 정답률 : 
#  [0.5144561  0.44386371 0.42552131 0.38419894 0.40748411]
# ExtraTreeRegressor 의 정답률 : 
#  [-0.12217783  0.00678632 -0.08384435 -0.21932571 -0.11496054]
# ExtraTreesRegressor 의 정답률 : 
#  [0.43250054 0.35576955 0.41246197 0.58150807 0.35665762]
# GammaRegressor 의 정답률 : 
#  [-0.02464224  0.00620062  0.00519296  0.00283814  0.00390519]
# GaussianProcessRegressor 의 정답률 : 
#  [-20.64725351 -21.49090215 -11.9492972  -13.81983289 -16.77055714]
# GeneralizedLinearRegressor 의 정답률 : 
#  [-0.00197936  0.0057909   0.00556097  0.00638608 -0.00768653]
# GradientBoostingRegressor 의 정답률 : 
#  [0.41064388 0.37652777 0.30174799 0.08157433 0.58884163]
# HistGradientBoostingRegressor 의 정답률 : 
#  [0.42778494 0.409817   0.29198039 0.39956425 0.35917319]
# HuberRegressor 의 정답률 : 
#  [0.48798469 0.51540423 0.4571698  0.3832262  0.50306708]
# IsotonicRegression 의 정답률 : 
#  [nan nan nan nan nan]
# KNeighborsRegressor 의 정답률 : 
#  [0.33964393 0.40830321 0.52850985 0.39339117 0.16972871]
# KernelRidge 의 정답률 : 
#  [-3.53167322 -2.89144609 -3.82976194 -3.75447339 -3.92905753]
# Lars 의 정답률 : 
#  [-8.00361229  0.33777914 -0.79513735 -5.77113161 -0.36581205]
# LarsCV 의 정답률 : 
#  [0.47865489 0.39806713 0.42933889 0.43411759 0.47732425]
# Lasso 의 정답률 : 
#  [0.33551819 0.254018   0.34897395 0.2440174  0.37814957]
# LassoCV 의 정답률 : 
#  [0.54106903 0.36511172 0.54612354 0.39599338 0.30576813]
# LassoLars 의 정답률 : 
#  [0.34294148 0.38955504 0.39194893 0.41567849 0.34072175]
# LassoLarsCV 의 정답률 : 
#  [0.39787178 0.3786296  0.51509584 0.6106868  0.35646157]
# LassoLarsIC 의 정답률 : 
#  [0.50386499 0.49130741 0.43878434 0.41283459 0.49976956]
# LinearRegression 의 정답률 : 
#  [0.45984668 0.35486527 0.49503038 0.48609013 0.50995508]
# LinearSVR 의 정답률 : 
#  [-0.19538698 -0.77631901 -0.2320376  -0.45890743 -0.81246327]
# MLPRegressor 의 정답률 : 
#  [-3.24416915 -3.069706   -3.05791905 -2.44288388 -2.58773079]
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
#  [0.14761728 0.14119768 0.12156778 0.12138881 0.10236069]
# OrthogonalMatchingPursuit 의 정답률 : 
#  [0.23021416 0.34700527 0.24094631 0.16993344 0.13068242]
# OrthogonalMatchingPursuitCV 의 정답률 : 
#  [0.38291073 0.39144125 0.46215112 0.48039332 0.50731232]
# PLSCanonical 의 정답률 : 
#  [-0.92897693 -1.69137758 -1.19713349 -1.16450954 -1.76744226]
# PLSRegression 의 정답률 : 
#  [0.54186227 0.36124876 0.46932135 0.55028577 0.34379704]
# PassiveAggressiveRegressor 의 정답률 : 
#  [0.51907964 0.34524511 0.31670717 0.41829025 0.53992997]
# PoissonRegressor 의 정답률 : 
#  [0.34753826 0.3193986  0.3568675  0.26301674 0.35115812]
# RANSACRegressor 의 정답률 : 
#  [ 0.14971171  0.12454663 -0.24693987  0.42838025  0.21312119]
# RadiusNeighborsRegressor 의 정답률 : 
#  [-0.0068094  -0.00041617 -0.00132734 -0.0007621  -0.00215966]
# RandomForestRegressor 의 정답률 : 
#  [0.32338387 0.35861994 0.46667917 0.54328675 0.34818983]
# RegressorChain 은 없는 모델
# Ridge 의 정답률 : 
#  [0.38454772 0.42757016 0.358136   0.37828405 0.38334808]
# RidgeCV 의 정답률 : 
#  [0.47007793 0.42525178 0.53682369 0.48250986 0.41694983]
# SGDRegressor 의 정답률 : 
#  [0.30169909 0.36236457 0.40574648 0.38901635 0.35667826]
# SVR 의 정답률 : 
#  [0.14633437 0.02940417 0.02082236 0.08907479 0.1440933 ]
# StackingRegressor 은 없는 모델
# TheilSenRegressor 의 정답률 : 
#  [0.53255155 0.3337448  0.44104345 0.53829298 0.45568437]
# TransformedTargetRegressor 의 정답률 : 
#  [0.61403485 0.4490517  0.48541883 0.36172953 0.37987313]
# TweedieRegressor 의 정답률 : 
#  [ 0.00611012 -0.01454483 -0.01841269 -0.06053251  0.00523712]
# VotingRegressor 은 없는 모델
# _SigmoidCalibration 의 정답률 : 
#  [nan nan nan nan nan]