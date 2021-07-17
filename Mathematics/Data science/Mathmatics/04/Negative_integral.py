import sympy
sympy.init_printing(use_latex='mathjax')
x = sympy.symbols('x')
f = x * sympy.exp(x) + sympy.exp(x)

print(f)    # x*exp(x) + exp(x)

print(sympy.integrate(f))   # x*exp(x)

x, y = sympy.symbols('x y')
f = 2 * x + y

print(f)    # 2*x + y

print(sympy.integrate(f, x))    # x**2 + x*y