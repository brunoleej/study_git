# PCA : 차원축소
# cumsum : 누적 합 계산 함수
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA

cancer = load_breast_cancer()
data = cancer.data
target = cancer.target
print(data.shape, target.shape) # (569, 30) (569,)

# pca = PCA(n_components=9)
# data2 = pca.fit_transform(data) 

# print(data2)
# print(data2.shape)            # (442, 7) 

# pca_EVR = pca.explained_variance_ratio_ # 컬럼의 변화율을 보여줌
# print(pca_EVR)
# print(sum(pca_EVR)) 

pca = PCA()
pca.fit(data)
cumsum = np.cumsum(pca.explained_variance_ratio_)   
print("cumsum : ", cumsum) 
# cumsum :  [0.98204467 0.99822116 0.99977867 0.9998996  0.99998788 0.99999453
#  0.99999854 0.99999936 0.99999971 0.99999989 0.99999996 0.99999998
#  0.99999999 0.99999999 1.         1.         1.         1.
#  1.         1.         1.         1.         1.         1.
#  1.         1.         1.         1.         1.         1.        ]

d = np.argmax(cumsum >= 0.99)+1
print("cumsum >= 0.95", cumsum > 0.99)
print("d : ", d)
# cumsum >= 0.95 [False  True  True  True  True  True  True  True  True  True  True  True
#   True  True  True  True  True  True  True  True  True  True  True  True
#   True  True  True  True  True  True]
# d :  2

import matplotlib.pyplot as plt
plt.plot(cumsum)
plt.grid()
plt.show()