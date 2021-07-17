import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris

iris = load_iris()

N = 10 # 앞의 10송이만 선택
X = iris.data[:N, :2] # 꽃받침 길이와 꽃받침 폭만 선택

plt.figure(figsize=(8, 8))

ax = sns.scatterplot(0, 1, data=pd.DataFrame(X), s=100, color=".2", marker="s")

for i in range(N):
    ax.text(X[i, 0] - 0.05, X[i, 1] + 0.03, "표본 {}".format(i + 1))

plt.xlabel("꽃받침 길이")
plt.ylabel("꽃받침 폭")

plt.title("붓꽃 크기 특성 (2차원 표시)")

plt.axis("equal")

plt.show()