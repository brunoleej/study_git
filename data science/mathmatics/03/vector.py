import numpy as np
import matplotlib.pylab as plt

plt.rc("font", size=18) # 그림의 폰트 크기를 18로 고정
gray = {"facecolor": "gray"}
black = {"facecolor": "black"}
red = {"facecolor": "red"}
green = {"facecolor": "green"}
blue = {"facecolor": "blue"}

a = np.array([1, 2])

plt.plot(0, 0, 'kP', ms=20)
plt.plot(a[0], a[1], 'ro', ms=20)

plt.annotate('', xy=[-0.6, 1.6], xytext=(0.2, 0.7), arrowprops=gray)
plt.annotate('', xy=a, xytext=(0, 0), arrowprops=black)
plt.annotate('', xy=a + [-1, 1], xytext=(-1, 1), arrowprops=black)

plt.text(0.35, 1.15, "$a$")
plt.text(1.15, 2.25, "$(1,2)$")
plt.text(-0.7, 2.1, "$a$")
plt.text(-0.9, 0.6, "평행이동")

plt.xticks(np.arange(-2, 4))
plt.yticks(np.arange(-1, 4))

plt.xlim(-2.4, 3.4)
plt.ylim(-0.8, 3.4)

plt.show()