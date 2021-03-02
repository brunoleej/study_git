import numpy as np

def heaviside_step(x):
    if isinstance(x, np.ndarray):
        return np.where(x >= 0, 1, 0)
    else:
        return 1.0 if x >= 0 else 0.0

print(heaviside_step(-0.0001), heaviside_step(0), heaviside_step(0.0001))   # 0.0 1.0 1.0