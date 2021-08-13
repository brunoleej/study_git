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
x1, x2 = point
point_storage.append(point)

def obj_func(x):
    x1, x2 = x
    return 1 + 2*x1 + 3*(x1**2+x2**2)+4*x1*x2

def gradient_obj_func(x):
    x1, x2 = x
    f_x1 = 6*x1 + 4*x2 + 2
    f_x2 = 4*x1 + 6*x2
    return np.array([f_x1, f_x2])

def gradient_alpha(alpha):
    print(point[0], point[1])
    value = -12*point[0] -8*point[1] + 3*(-8*point[0] - 12*point[1])\
    *(-alpha*(4*point[0] + 6*point[1])+point[1]) + (-4*point[0] - 6*point[1])\
    *(-4*alpha*(6*point[0] + 4*point[1] + 2) + 4*point[0])\
    + (-alpha*(4*point[0] + 6*point[1]) + point[1])*(-24*point[0]-16*point[1] -8)\
    + 3*(-alpha*(6*point[0]+4*point[1] + 2)+point[0])*(-12*point[0]-8*point[1]-4)-4
    return value

# steepest Descent
i = 0
while True:
    i += 1
    print(point)
    gradient = gradient_obj_func(point)

    step_size = optimize.newton(gradient_alpha, 0)
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

'''
Prespecified threshold : 0.0003
Convergence Value : 0.00023289679098294767
Convergence at 12 iterations
'''