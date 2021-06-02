# relu 함수

import numpy as np
import matplotlib.pyplot as plt

def relu(x) :
    return np.maximum(0, x) # 0이하의 값들은 0으로, 그 이상의 값들은 x로 반환

x = np.arange(-5, 5, 0.1)
y = relu(x)

print(x)
print(y)

# 시각화
plt.plot(x, y)
plt.grid()
plt.show()

