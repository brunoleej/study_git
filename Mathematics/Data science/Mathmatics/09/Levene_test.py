import scipy as sp
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

N1 = 100
N2 = 100
sigma_1 = 1
sigma_2 = 1.2
np.random.seed(0)

x1 = sp.stats.norm(0, sigma_1).rvs(N1)
x2 = sp.stats.norm(0, sigma_2).rvs(N2)
ax = sns.distplot(x1, kde=False, fit=sp.stats.norm, label="1번 데이터 집합")
ax = sns.distplot(x2, kde=False, fit=sp.stats.norm, label="2번 데이터 집합")
ax.lines[0].set_linestyle(":")
plt.legend()
plt.show()

print(x1.std(), x2.std())   # 1.0078822447165796 1.2416003969261071
print(sp.stats.bartlett(x1, x2))    # BartlettResult(statistic=4.253473837232266, pvalue=0.039170128783651344)
print(sp.stats.fligner(x1, x2)) # FlignerResult(statistic=7.224841990409457, pvalue=0.007190150106748367)
print(sp.stats.levene(x1, x2))  # LeveneResult(statistic=7.680708947679437, pvalue=0.0061135154970207925)
