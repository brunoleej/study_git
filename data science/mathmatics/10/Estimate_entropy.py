import scipy as sp
import scipy.stats

p = [0.5, 0.5]
print(sp.stats.entropy(p, base=2))  # 1.0
