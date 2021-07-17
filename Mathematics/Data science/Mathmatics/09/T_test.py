import numpy as np
import scipy as sp
import scipy.stats

N = 10
mu_0 = 0
np.random.seed(0)
x = sp.stats.norm(mu_0).rvs(N)
print(sp.stats.ttest_1samp(x, popmean=0))