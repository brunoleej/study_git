# elu 함수

import numpy as np
import matplotlib.pyplot as plt

alpa = 0.5
def elu(x) :
    # return  np.maximum(2*(np.exp(x)-1) * abs(x)/-x, x) # relu와 유사함, 입력이 0 이하일 경우 부드럽게 깎아준다.
    return (x>=0)*x + (x<0)*alpa*(np.exp(x)-1)  # 0보다 크면 x를 반환, 0보다 작으면 뒤의 계산식 반환

x = np.arange(-5, 5, 0.1)
y = elu(x)

print(x)
print(y)

# 시각화
plt.plot(x, y)
plt.grid()
plt.show()
