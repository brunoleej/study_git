# LeakyReLU 함수

import numpy as np
import matplotlib.pyplot as plt

def lekrelu(x) :
    return np.maximum(0.01*x, x) # relu와 유사함, 입력 값이 음수일 때 완만한 선형 함수를 그려준다.

x = np.arange(-5, 5, 0.1)
y = lekrelu(x)

print(x)
print(y)

# 시각화
plt.plot(x, y)
plt.grid()
plt.show()

