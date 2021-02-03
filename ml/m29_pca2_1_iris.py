# PCA : 차원축소, 컬럼 재구성
# cumsum : 배열에서 주어진 축에 따라 누적되는 원소들의 누적 합을 계산하는 함수.

import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

datasets = load_iris()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (150, 4) (150,)

# pca = PCA(n_components=9)
# x2 = pca.fit_transform(x)  # fit_transform : 전처리 fit과 transform 한꺼번에 한다.

# print(x2)
# print(x2.shape)            # (442, 7) >> 컬럼을 압축시켰다. 컬럼 재구성됨

# pca_EVR = pca.explained_variance_ratio_ # 컬럼이 어느 정도의 변화율을 보여주었는지 보여준다.
# print(pca_EVR)
# print(sum(pca_EVR)) 
# n_components=7개 압축률 : 0.9479436357350414
# n_components=8개 압축률 : 0.9913119559917797
# n_components=9개 압축률 : 0.9991439470098977

pca = PCA()
pca.fit(x)
cumsum = np.cumsum(pca.explained_variance_ratio_)   
print("cumsum : ", cumsum)  # cumsum 누적 합을 계산
# cumsum :  [0.92461872 0.97768521 0.99478782 1.        ]

d = np.argmax(cumsum >= 0.95)+1 # cumsum이 0.95 이상인 컬럼을 True 로 만든다.
print("cumsum >= 0.95", cumsum > 0.95)
print("d : ", d)
# cumsum >= 0.95 [False  True  True  True]
# d :  2

import matplotlib.pyplot as plt
plt.plot(cumsum)
plt.grid()
plt.show()