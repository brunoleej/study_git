# beta distribution
import numpy as np
import scipy as sp

np.random.seed(0)
x = sp.stats.beta(15, 12).rvs(10000)
sns.distplot(x, kde=False, norm_hist=True)
plt.title("베타 분포를 따르는 표본의 히스토그램")
plt.show()