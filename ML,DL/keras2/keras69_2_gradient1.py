# 이차함수 그래프를 그린다.

import numpy as np
import matplotlib.pyplot as plt

f = lambda x : x**2 - 4*x + 6   # 이차함수
x = np.linspace(-1, 6, 100)     # -1부터 6까지 100개의 요소를 1차원 배열로 만든다.
y = f(x)

# 시각화
plt.plot(x, y, 'k-')    # 'k-' 색깔 지정
plt.plot(2, 2, 'sk')    # (2,2) 에 해당하는 지점에 점을 찍는다.
# plt.plot(2, 2, 'bx')    # (2,2) 에 해당하는 지점에 blue색 x을 찍는다.
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
plt.show()
