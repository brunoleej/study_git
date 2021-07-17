import numpy as np

print(-0.5 * np.log2(0.5) - 0.5 * np.log2(0.5)) # 1.0
print(-0.8 * np.log2(0.8) - 0.2 * np.log2(0.2)) # 0.7219280948873623

eps = np.finfo(float).eps
print(-1 * np.log2(1) - eps * np.log2(eps)) # 1.1546319456101628e-14