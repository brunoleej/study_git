import numpy as np
import matplotlib.pylab as plt

a = np.array([2, 2])
b = np.array([3, 4])
c = np.array([4, 1])
d = a + (c - a)
e = b + (c - a)

plt.rc("font", size=18) # 그림의 폰트 크기를 18로 고정
gray = {"facecolor": "gray"}
black = {"facecolor": "black"}
red = {"facecolor": "red"}
green = {"facecolor": "green"}
blue = {"facecolor": "blue"}

plt.annotate('', xy=b, xytext=a, arrowprops=black)
plt.annotate('', xy=e, xytext=d, arrowprops=black)
plt.annotate('', xy=c, xytext=[0, 0], arrowprops=gray)

plt.plot(0, 0, 'kP', ms=10)
plt.plot(a[0], a[1], 'ro', ms=10)
plt.plot(b[0], b[1], 'ro', ms=10)
plt.plot(c[0], c[1], 'ro', ms=10)

plt.text(1.6, 1.5, "서울")
plt.text(2.5, 4.3, "한국")
plt.text(3.5, 0.5, "도쿄")
plt.text(4.9, 3.2, "일본")

plt.xticks(np.arange(-2, 7))
plt.yticks(np.arange(-1, 6))

plt.xlim(-1.4, 6.4)
plt.ylim(-0.6, 5.8)

plt.show()