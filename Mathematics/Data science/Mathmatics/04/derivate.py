from scipy.misc import derivative

def f(x):
     return x**3 - 3 * x**2 + x

print(derivative(f, 0, dx=1e-6))    # 1.000000000001
print(derivative(f, 1, dx=1e-6))    # -2.000000000002
