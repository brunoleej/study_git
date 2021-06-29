import numpy as np

x = np.random.randint(1, 100, size=10)
print(x)    # [91 26 75 50 82 46 82 98 66  5]

even_mask = x % 2 == 0
print(even_mask)    # [False  True False  True  True  True  True  True  True False]

# boolean index
print(x[even_mask])     # [78 20 52 90 28 28]
print(x[x % 2 == 0])    # [24 16 38 64 62 22]
print(x[x > 30])        # [66 56 82 63 56 74 87 95 68]


# Multiconditional
# & : AND
# | : OR
print(x[(x < 30) | (x > 50)])   # [96 71 75 52 96 60]