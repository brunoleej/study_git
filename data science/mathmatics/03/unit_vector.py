import numpy as np

a = np.array([1, 0])
b = np.array([0, 1])
c = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
print(np.linalg.norm(a), np.linalg.norm(b), np.linalg.norm(c))  # 1.0 1.0 0.9999999999999999\

