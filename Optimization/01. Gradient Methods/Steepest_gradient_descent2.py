import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
from matplotlib import cm

# Hyperparameter
threshold = 0.0003     # convergence threshold
# step_size = 0.01     # step-size factor
point_storage = []     # point history

# initial point
point = np.array([-2.0, 2.0])   
x, y = point
point_storage.append(point)

def obj_func(x):
    x, y= x
    return 3*(x**2 + y**2) + 4*x*y + 5*x + 6*y + 7

def gradient_obj_func(x):
    x, y = x
    f_x = 6*x + 4*y + 5
    f_y = 4*x + 6*y + 6
    return np.array([f_x, f_y])

def gradient_alpha(alpha):
    value = -54*point[0] - 56*point[1] + (-alpha*(4*point[0] + 6*point[1] + 6) + point[1])*(-24*point[0] - 16*point[1] - 20) + 3*(-alpha*(4*point[0] + 6*point[1] + 6) + point[1])*(-8*point[0] - 12*point[1] - 12) + (-4*alpha*(6*point[0] + 4*point[1] + 5) + 4*point[0])*(-4*point[0] - 6*point[1] - 6) + 3*(-alpha*(6*point[0] + 4*point[1] + 5) + point[0])*(-12*point[0] - 8*point[1] - 10) - 61
    return value

# steepest Descent
i = 0
while True:
    i += 1
    print(point)
    gradient = gradient_obj_func(point)

    step_size = optimize.newton(gradient_alpha, 0, maxiter = 4000)
    point = point - step_size * gradient_obj_func(point)
    point_storage.append(point)

    print("Point History : {}".format(point_storage))
    print("Next Point : {}".format(point))
    print('Present step_size : {}\n'.format(step_size))
    compare_value = np.linalg.norm(point_storage[i] - point_storage[i-1])

    if compare_value < threshold:
        print("Convergence Value : {}".format(compare_value))
        print("Convergence at {} iterations".format(i))
        break
                
# plotting 3d graph
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')
title = ax.set_title('objective_function')

x1 = np.arange(-2.5, 2.5, 0.1)
x2 = np.arange(-2.5, 2.5, 0.1)

X1, X2 = np.meshgrid(x1, x2)
graph = ax.plot_surface(X1, X2, obj_func((X1, X2)),cmap=cm.coolwarm, alpha=.5)

point_storage_np = np.array(point_storage)
computed_x1 = point_storage_np[:, 0]
computed_x2 = point_storage_np[:, 1]
computed_z = obj_func((computed_x1, computed_x2))
graph2 = ax.scatter(computed_x1, computed_x2, computed_z, color='black', linewidth = 0.01)

plt.show()
