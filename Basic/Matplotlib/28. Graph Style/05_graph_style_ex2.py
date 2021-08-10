import matplotlib.pyplot as plt

plt.style.use('default')
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
# plt.rcParams['xtick.major.size'] = 7
# plt.rcParams['ytick.major.size'] = 7
# plt.rcParams['xtick.minor.visible'] = True
# plt.rcParams['ytick.minor.visible'] = True

plt.plot([1, 2, 3, 4], [4, 6, 2, 7])
plt.show()