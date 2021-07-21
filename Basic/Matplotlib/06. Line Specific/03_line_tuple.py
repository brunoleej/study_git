import matplotlib.pyplot as plt

plt.plot([1, 2, 3], [4, 4, 4], linestyle=(0, (1, 1)), color='C0', label='(0, (1, 1))')
plt.plot([1, 2, 3], [3, 3, 3], linestyle=(0, (1, 5)), color='C0', label='(0, (1, 5))')
plt.plot([1, 2, 3], [2, 2, 2], linestyle=(0, (5, 1)), color='C0', label='(0, (5, 1))')
plt.plot([1, 2, 3], [1, 1, 1], linestyle=(0, (3, 5, 1, 5)), color='C0', label='(0, (3, 5, 1, 5))')
plt.xlabel('X-Axis')
plt.ylabel('Y-Axis')
plt.axis([0.8, 3.2, 0.5, 5.0])
plt.legend(loc='upper right', ncol=2)
plt.tight_layout()
plt.show()