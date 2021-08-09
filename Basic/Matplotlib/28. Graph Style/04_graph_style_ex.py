import matplotlib.pyplot as plt

plt.style.use('default')
plt.rcParams['lines.linewidth'] = 3
plt.rcParams['lines.linestyle'] = '-'
# plt.rcParams['lines.linewidth'] = 5
# plt.rcParams['lines.linestyle'] = '--'

plt.plot([1, 2, 3, 4], [4, 6, 2, 7])
plt.show()