# PCA : 차원축소
# cumsum : 누적 합 계산 함수
import numpy as np
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA

wine = load_wine()
data = wine.data
target = wine.target
print(data.shape, target.shape) # (178, 13) (178,)

# pca = PCA(n_components=9)
# data2 = pca.fit_transform(data)  

# print(data2)
# print(data2.shape)            # (442, 7) 

# pca_EVR = pca.explained_variance_ratio_ # 컬럼이 어느 정도의 변화율을 보여주었는지 보여준다.
# print(pca_EVR)
# print(sum(pca_EVR)) 

pca = PCA()
pca.fit(x)
cumsum = np.cumsum(pca.explained_variance_ratio_)   
print("cumsum : ", cumsum)  # cumsum 누적 합을 계산
# cumsum :  [0.99809123 0.99982715 0.99992211 0.99997232 0.99998469 0.99999315
#  0.99999596 0.99999748 0.99999861 0.99999933 0.99999971 0.99999992
#  1.        ]

d = np.argmax(cumsum >= 0.99)+1
print("cumsum >= 0.99", cumsum > 0.99)
print("d : ", d)
# cumsum >= 0.99 [ True  True  True  True  True  True  True  True  True  True  True  True  True]
# d :  1

import matplotlib.pyplot as plt
plt.plot(cumsum)
plt.grid()
plt.show()