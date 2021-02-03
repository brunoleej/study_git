# PCA: 차원축소
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.decomposition import PCA

diabetes = load_diabetes()
data = diabetes.data
target = diabetes.target
# print(data.shape, target.shape) # (442, 10) (442,)

pca = PCA(n_components=8)  # n_components: 축소할 차원 명시
data2 = pca.fit_transform(data) 

pca_EVR = pca.explained_variance_ratio_ # 컬럼의 변화율을 표시
print(pca_EVR)
print(sum(pca_EVR)) 
# n_components=7개 압축률 : 0.9479436357350414
# n_components=8개 압축률 : 0.9913119559917797
# n_components=9개 압축률 : 0.9991439470098977 