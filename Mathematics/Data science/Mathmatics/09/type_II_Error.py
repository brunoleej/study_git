import numpy as np
import scipy as sp
import scipy.stats

N_1 = 50
mu_1 = 0
sigma_1 = 1
N_2 = 100
mu_2 = 0.5
sigma_2 = 1
np.random.seed(0)
x1 = sp.stats.norm(mu_1, sigma_1).rvs(N_1)
x2 = sp.stats.norm(mu_2, sigma_2).rvs(N_2)

print(sp.stats.ttest_ind(x1, x2, equal_var=True))
# Ttest_indResult(statistic=-2.6826951236616963, pvalue=0.008133970915722658)