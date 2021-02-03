# PCA : 차원축소
# cumsum : 누적 합 계산 함수
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.decomposition import PCA

diabetes = load_diabetes()
data = diabetes.data
target = diabetes.target
print(data.shape, target.shape) # (442, 10) (442,)

# pca = PCA(n_components=9)
# data2 = pca.fit_transform(data)  

# print(data2)
# print(data2.shape)            # (442, 7) 

# pca_EVR = pca.explained_variance_ratio_ # 컬럼의 변화율을 보여줌
# print(pca_EVR)
# print(pca_EVR)
# print(sum(pca_EVR)) 

pca = PCA()
pca.fit(data)
cumsum = np.cumsum(pca.explained_variance_ratio_)   
print("cumsum : ", cumsum)  
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