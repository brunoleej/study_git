# PCA : 차원축소, 컬럼 재구성
# cumsum : 배열에서 주어진 축에 따라 누적되는 원소들의 누적 합을 계산하는 함수.

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.decomposition import PCA

datasets = load_diabetes()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (442, 10) (442,)

# pca = PCA(n_components=9)
# x2 = pca.fit_transform(x)  # fit_transform : 전처리 fit과 transform 한꺼번에 한다.

# print(x2)
# print(x2.shape)            # (442, 7) >> 컬럼을 압축시켰다. 컬럼 재구성됨

# pca_EVR = pca.explained_variance_ratio_ # 컬럼이 어느 정도의 변화율을 보여주었는지 보여준다.
# print(pca_EVR)
# print(sum(pca_EVR)) 

pca = PCA()
pca.fit(x)
cumsum = np.cumsum(pca.explained_variance_ratio_)   
print("cumsum : ", cumsum)  # cumsum 누적 합을 계산
# cumsum :  [0.40242142 0.55165324 0.67224947 0.76779711 0.83401567 0.89428759
#  0.94794364 0.99131196 0.99914395 1.        ]

d = np.argmax(cumsum >= 0.95)+1
print("cumsum >= 0.95", cumsum > 0.95)
print("d : ", d)
# cumsum >= 0.95 [False False False False False False False  True  True  True]
# d :  8

import matplotlib.pyplot as plt
plt.plot(cumsum)
plt.grid()
plt.show()

