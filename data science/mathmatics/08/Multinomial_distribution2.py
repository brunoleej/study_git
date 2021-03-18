import scipy as sp
import numpy as np
import matplotlib.pyplot as plt

N = 30
mu = [0.1, 0.1, 0.1, 0.1, 0.3, 0.3]
rv = sp.stats.multinomial(N, mu)
np.random.seed(0)
X = rv.rvs(100)
X[:10]

plt.boxplot(X)
plt.title("다항분포의 시뮬레이션 결과")
plt.xlabel("클래스")
plt.ylabel("표본값")
plt.show()