import numpy as np
import matplotlib.pyplot as plt

x = rv.rvs(20)
x_sorted = np.sort(x)
print(x_sorted)
'''
array([-0.97727788, -0.85409574, -0.20515826, -0.15135721, -0.10321885,
 0.12167502, 0.14404357, 0.3130677 , 0.33367433, 0.40015721,
 0.4105985 , 0.44386323, 0.76103773, 0.95008842, 0.97873798,
 1.45427351, 1.49407907, 1.76405235, 1.86755799, 2.2408932 ])
'''

from scipy.stats.morestats import _calc_uniform_order_statistic_medians
position = _calc_uniform_order_statistic_medians(len(x))
print(position)
'''
array([0.03406367, 0.08261724, 0.13172109, 0.18082494, 0.2299288 ,
 0.27903265, 0.32813651, 0.37724036, 0.42634422, 0.47544807,
 0.52455193, 0.57365578, 0.62275964, 0.67186349, 0.72096735,
 0.7700712 , 0.81917506, 0.86827891, 0.91738276, 0.96593633])
'''

qf = rv.ppf(position)
print(qf)
'''
array([-0.8241636 , -0.38768012, -0.11829229, 0.08777425, 0.26091865,
 0.4142824 , 0.55493533, 0.68726332, 0.81431072, 0.93841854,
 1.06158146, 1.18568928, 1.31273668, 1.44506467, 1.5857176 ,
 1.73908135, 1.91222575, 2.11829229, 2.38768012, 2.8241636 ])
'''

np.random.seed(0)
plt.figure(figsize=(7, 7))
sp.stats.probplot(x, plot=plt)
plt.axis("equal")
plt.show()