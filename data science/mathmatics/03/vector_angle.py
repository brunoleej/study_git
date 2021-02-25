import numpy as np
import matplotlib.pyplot as plt

plt.plot([0, 1], [0, 2], 'k-', lw=3)
plt.plot([0, 1], [0, 0], 'k-', lw=3)
plt.plot([1, 1], [0, 2], 'k-', lw=3)

plt.text(0.05, 1, "빗변 h")
plt.text(0.35, -0.2, "밑변 b")
plt.text(1.05, 1, "높이 a")
plt.text(0.12, 0.06, r"$\theta$")

plt.xticks(np.arange(-2, 4))
plt.yticks(np.arange(-1, 4))

plt.xlim(-1.1, 2.1)
plt.ylim(-0.5, 2.3)

plt.show()
