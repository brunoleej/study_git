import scipy as sp
import scipy.stats
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

rv = sp.stats.norm(loc=1, scale=2)

rv.rvs(size=(3, 5), random_state=0)

sns.distplot(rv.rvs(size=10000, random_state=0))
plt.title("랜덤 표본 생성 결과")
plt.xlabel("표본값")
plt.ylabel("count")
plt.xlim(-8, 8)
plt.show()