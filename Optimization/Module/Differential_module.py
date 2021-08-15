from sympy import *
import numpy as np

# x = Symbol('x', integer = True)
# x = Symbol('x', real = True)
# x = Symbol('x', complex = True)
# x = Symbol('x', positive = True)
x1 = Symbol('x1')
x2 = Symbol('x2')
alpha = Symbol('a')

# Initial point
init_point = [x1,x2]

# objective function
def obj_func(x):
    x1, x2 = x
    return 1 + 2*x1 + 3*(x1**2 + x2**2) + 3*x1*x2

# partial-derivate function
def g(x):
    x1, x2 = x
    f_x1 = diff(obj_func(init_point), x1)
    f_x2 = diff(obj_func(init_point), x2)
    return np.array([f_x1, f_x2])

# calculate step_size alpha function
def calc_step(alpha):
    step = obj_func(init_point - alpha * g(init_point))
    step_diff = diff(step, alpha)
    string_step = str(step_diff)
    string_step = string_step.replace('a', 'alpha')
    string_step = string_step.replace('x1', 'point[0]')
    string_step = string_step.replace('x2', 'point[1]')
    return string_step

print(calc_step(alpha))