import matplotlib.pyplot as plt

plt.plot([1, 2, 3, 4], [1, 4, 9, 16])
plt.xlabel('X-Label')
plt.ylabel('Y-Label')
plt.text(1, 15, r'$\sin (x) \ \cos (x) \ \tan (x)$', fontdict={'size': 16})
plt.text(1, 12, r'$\lim_{x\rightarrow 2} (x^2 - x + 2)$', fontdict={'size': 16})
plt.text(1, 8, r'$\sum_{n=0}^{10}{(n^2 + n)}$', fontdict={'size': 16})
plt.show()