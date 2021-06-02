import pandas as pd
import numpy as np

#1. DATA
wine = pd.read_csv('../data/csv/winequality-white.csv', header=0, sep=';',index_col=None)
# print(wine.head())

print(np.unique(wine['quality']))   # [3 4 5 6 7 8 9]

count_data = wine.groupby('quality')['quality'].count()
# groupby : 
print(count_data)
'''
quality
3      20
4     163
5    1457
6    2198
7     880
8     175
9       5
'''
# 문제점 : 카테고리는 많은데 y가 5, 6, 7에 모여있다. 
# 해결점 : 카테고리를 줄여본다.(상, 중, 하) >> acc를 높일 수 있다. ***************
# 주의점 : 카테고리를 변경할 수 있는 권한이 있을 때만 카테고리 수를 조절할 수 있다.


import matplotlib.pyplot as plt
count_data.plot()
plt.show()
