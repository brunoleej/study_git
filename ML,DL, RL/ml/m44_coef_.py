# coef_ : weight
# intercept_ : bias  

x = [-3, 31, -11, 4, 0, 22, -2, -5, -25, -14]
y = [-2, 32, -10, 5, 1, 23, -1, -4, -24, -13] 
# weight : 1, bias : 1
# y = [-5, 63, -21, 9, 1, 45, -3, -9, -49, -27]
# weight : 2, bias : 1
# y = [-3, 65, -19, 11, 3, 47, -1, -7, -47, -25]
# weight : 2, bias : 3

print(x, "\n", y)

import matplotlib.pyplot as plt
plt.plot(x, y)
# plt.show()

import pandas as pd
df = pd.DataFrame({'X':x, 'Y':y})
print(df)
print(df.shape) # (10, 2)

x_train = df.loc[:,'X']
y_train = df.loc[:,'Y']

print(x_train.shape, y_train.shape) # (10,) (10,)

print(type(x_train))    # <class 'pandas.core.series.Series'>

x_train = x_train.values.reshape(len(x_train), 1)   # (10, 1) : 모델을 구성하기 위해서 x를 스칼라 형태로 만든다.
print(x_train.shape, y_train.shape) # (10, 1) (10,)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train, y_train)

score = model.score(x_train, y_train)
print("score ", score)    # 1.0

print("Weight : ",model.coef_)        # [1.]
print("Bias : ", model.intercept_)    # 1.0
