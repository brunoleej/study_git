import scipy as sp
import numpy as np
import matplotlib.pyplot as plt

N = 30
mu = [0.1, 0.1, 0.1, 0.1, 0.3, 0.3]
rv = sp.stats.multinomial(N, mu)
np.random.seed(0)
X = rv.rvs(100)
X[:10]

df = pd.DataFrame(X).stack().reset_index()
df.columns = ["시도", "클래스", "표본값"]
sns.violinplot(x="클래스", y="표본값", data=df, inner="quartile")
sns.swarmplot(x="클래스", y="표본값", data=df, color=".3")
plt.title("다항분포의 시뮬레이션 결과")
plt.show()