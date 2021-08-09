import matplotlib.pyplot as plt

plt.plot([1, 2, 3, 4], [1, 4, 9, 16])
plt.xlabel('X-Label')
plt.ylabel('Y-Label')
plt.text(1, 15, r'$\acute a, \bar a, \tilde a$', fontdict={'size': 16})
plt.text(1, 13, r'$\vec a \cdot \vec a = |\vec a|^2$', fontdict={'size': 16})
plt.text(1, 11, r'$\overline{abc}$', fontdict={'size': 16})

plt.show()