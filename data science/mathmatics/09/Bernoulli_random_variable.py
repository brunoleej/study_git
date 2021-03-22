import numpy as np
import scipy as sp
import scipy.stats

N = 10
mu_0 = 0.5
np.random.seed(0)
x = sp.stats.bernoulli(mu_0).rvs(N)
n = np.count_nonzero(x)

print(n)    # 7
print(sp.stats.binom_test(n, N))    # 0.3437499999999999