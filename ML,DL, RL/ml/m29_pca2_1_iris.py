# PCA : 차원축소
# cumsum : 누적 합 계산 함수
import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

iris = load_iris()
data = iris.data
target = iris.target
print(data.shape, target.shape) # (150, 4) (150,)

# pca = PCA(n_components=9)
# data2 = pca.fit_transform(data)  

# print(data2)
# print(data2.shape)            # (442, 7)

# pca_EVR = pca.explained_variance_ratio_ # 컬럼의 변화율을 보여줌
# print(pca_EVR)
# print(sum(pca_EVR)) 
# n_components=7개 압축률 : 0.9479436357350414
# n_components=8개 압축률 : 0.9913119559917797
# n_components=9개 압축률 : 0.9991439470098977

pca = PCA()
pca.fit(data)
cumsum = np.cumsum(pca.explained_variance_ratio_)   
print("cumsum: ", cumsum)  
# cumsum :  [0.92461872 0.97768521 0.99478782 1.        ]

d = np.argmax(cumsum >= 0.95)+1 # cumsum이 0.95 이상인 컬럼을 True 로 만듬
print("cumsum >= 0.95", cumsum > 0.95)
print("d : ", d)
# cumsum >= 0.95 [False  True  True  True]
# d :  2

import matplotlib.pyplot as plt
plt.plot(cumsum)
plt.grid()
plt.show()