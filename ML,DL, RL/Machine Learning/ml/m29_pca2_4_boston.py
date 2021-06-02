# PCA : 차원축소
# cumsum : 누적 합 계산 함수
import numpy as np
from sklearn.datasets import load_boston
from sklearn.decomposition import PCA

boston = load_boston()
data = boston.data
target = boston.target
print(data.shape, target.shape) # (506, 13) (506,)

# pca = PCA(n_components=9)
# data2 = pca.fit_transform(data) 

# print(x2)
# print(x2.shape)            # (442, 7) 

# pca_EVR = pca.explained_variance_ratio_ # 컬럼의 변화율을 보여줌
# print(pca_EVR)
# print(sum(pca_EVR)) 

pca = PCA()
pca.fit(data)
cumsum = np.cumsum(pca.explained_variance_ratio_)   
print("cumsum : ", cumsum)  
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