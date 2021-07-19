import matplotlib.pyplot as plt

plt.plot([1, 2, 3, 4], [2, 3, 5, 10], label='Price ($)')
plt.xlabel('X-Axis')
plt.ylabel('Y-Axis')
# plt.legend(loc=(0.0, 0.0))      # under
plt.legend(loc=(0.5, 0.5))      # middle
# plt.legend(loc=(1.0, 1.0))    # top

plt.show()