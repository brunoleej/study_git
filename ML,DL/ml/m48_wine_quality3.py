import pandas as pd
import numpy as np

#1. DATA
wine = pd.read_csv('../data/csv/winequality-white.csv', header=0, sep=';',index_col=None)
# print(wine.head())
print(wine.shape)       # (4898, 12)
# print(wine.describe())
'''
       fixed acidity  volatile acidity  citric acid  residual sugar    chlorides  free sulfur dioxide  total sulfur dioxide      density           pH    sulphates      alcohol      quality
count    4898.000000       4898.000000  4898.000000     4898.000000  4898.000000          4898.000000           4898.000000  4898.000000  4898.000000  4898.000000  4898.000000  4898.000000
mean        6.854788          0.278241     0.334192        6.391415     0.045772            35.308085            138.360657     0.994027     3.188267     0.489847    10.514267     5.877909
std         0.843868          0.100795     0.121020        5.072058     0.021848            17.007137             42.498065     0.002991     0.151001     0.114126     1.230621     0.885639
min         3.800000          0.080000     0.000000        0.600000     0.009000             2.000000              9.000000     0.987110     2.720000     0.220000     8.000000     3.000000
25%         6.300000          0.210000     0.270000        1.700000     0.036000            23.000000            108.000000     0.991723     3.090000     0.410000     9.500000     5.000000
50%         6.800000          0.260000     0.320000        5.200000     0.043000            34.000000            134.000000     0.993740     3.180000     0.470000    10.400000     6.000000
75%         7.300000          0.320000     0.390000        9.900000     0.050000            46.000000            167.000000     0.996100     3.280000     0.550000    11.400000     6.000000
max        14.200000          1.100000     1.660000       65.800000     0.346000           289.000000            440.000000     1.038980     3.820000     1.080000    14.200000     9.000000
'''

wine_npy = wine.values

# x = wine_npy[:,:11]
# y = wine_npy[:,11]
# print(x.shape, y.shape) # (4898, 11) (4898,)

y = wine['quality']
x = wine.drop('quality', axis=1)
print(x.shape, y.shape) # (4898, 11) (4898,)

print(np.unique(y)) # [3 4 5 6 7 8 9]

newlist = []
# y 카테고리를 3개로 줄여준다. (카테고리 나누는 기준은 알아서 정할 수 있다.)
for i in list(y) :
    if i <=4 : 
        newlist += [0]
    elif i <=7 :
        newlist += [1]
    else :
        newlist += [2]
y = newlist
print(np.unique(y)) # [0 1 2]

# preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=66, shuffle=True)

scale = StandardScaler()
scale.fit(x_train)
x_train = scale.transform(x_train)
x_test = scale.transform(x_test)

print(x_train.shape, x_test.shape)  # (3918, 11) (980, 11)


#2. Modeling
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# model = KNeighborsClassifier()
model = RandomForestClassifier()
# model = XGBClassifier()

model.fit(x_train, y_train)
score = model.score(x_test, y_test)
print("score : ", score)

# KNeighborsClassifier
# score :  0.5663265306122449

# RandomForestClassifier
# score :  0.7173469387755103

# XGBClassifier
# score :  0.6816326530612244

# y 카테고리 조절한 후, RandomForestClassifier 
# score :  0.9479591836734694
