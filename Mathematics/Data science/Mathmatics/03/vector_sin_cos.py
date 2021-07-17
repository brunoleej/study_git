import numpy as np
import matplotlib.pylab as plt

x = np.linspace(0, np.pi/2, 100)
y1 = np.sin(x)
y2 = np.cos(x)

plt.plot(x, y1, 'r--', lw=3, label=r"$\sin\theta$")
plt.plot(x, y2, 'b-', lw=3, label=r"$\cos\theta$")

plt.legend()

plt.xticks([0, np.pi/4, np.pi/2], [r'$0^{\circ}$', r'$45^{\circ}$', r'$90^{\circ}$'])

plt.xlabel(r"$\theta$")
plt.title(r"$\sin\theta$와 $\cos\theta$의 그래프")

plt.show()
