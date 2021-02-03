# PCA : 차원축소, 컬럼 재구성
# cumsum : 배열에서 주어진 축에 따라 누적되는 원소들의 누적 합을 계산하는 함수.

import numpy as np
from sklearn.datasets import load_boston
from sklearn.decomposition import PCA

datasets = load_boston()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (506, 13) (506,)


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
# cumsum :  [0.80582318 0.96887514 0.99022375 0.99718074 0.99848069 0.99920791
#  0.99962696 0.9998755  0.99996089 0.9999917  0.99999835 0.99999992
#  1.        ]

d = np.argmax(cumsum >= 0.99)+1
print("cumsum >= 0.95", cumsum > 0.99)
print("d : ", d)
# cumsum >= 0.95 [False False  True  True  True  True  True  True  True  True  True  True  True]
# d :  3

import matplotlib.pyplot as plt
plt.plot(cumsum)
plt.grid()
plt.show()

