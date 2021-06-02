# tanh 함수

import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5, 5, 0.1)
y = np.tanh(x)              # 값이 -1과 1 사이로 수렴된다.

# 시각화
plt.plot(x, y)
plt.grid()
plt.show()
