import sympy
# Juypter 노트북에서 수학식의 LaTeX 표현을 위해 필요함
sympy.init_printing(use_latex='mathjax')

x = sympy.symbols('x')
print(x)

print(type(x))  # <class 'sympy.core.symbol.Symbol'>

f = x * sympy.exp(x)
print(f)    # x*exp(x)

print(sympy.diff(f))    # x*exp(x) + exp(x)

print(sympy.simplify(sympy.diff(f)))    # (x + 1)*exp(x)

x, y = sympy.symbols('x y')
f = x ** 2 + 4 * x * y + 4 * y ** 2

print(f)    # x**2 + 4*x*y + 4*y**2

print(sympy.diff(f, x)) # 2*x + 4*y

print(sympy.diff(f, y)) # 4*x + 8*y

x, mu, sigma = sympy.symbols('x mu sigma')
f = sympy.exp((x - mu) ** 2 / sigma ** 2)

print(f)    # exp((-mu + x)**2/sigma**2)

print(sympy.diff(f, x)) # (-2*mu + 2*x)*exp((-mu + x)**2/sigma**2)/sigma**2

print(sympy.simplify(sympy.diff(f, x))) # 2*(-mu + x)*exp((mu - x)**2/sigma**2)/sigma**2

print(sympy.diff(f, x, x))  # 2*(1 + 2*(mu - x)**2/sigma**2)*exp((mu - x)**2/sigma**2)/sigma**2
