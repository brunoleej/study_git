import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# Hyperparameter
threshold = 0.0003   # convergence threshold
step_size = 0.01     # step-size factor
point_storage = []   # point history


# initial point
point = np.array([-2.0, 2.0])
x, y = point  # decision variable
point_storage.append(point)

def obj_func(x):
    x, y = x 
    f_x = 3*(x**2 + y**2) + 4*x*y + 5*x + 6*y + 7
    return f_x

def g(x):
    x, y = x
    f_x = 6*x + 4*y + 5
    f_y = 4*x + 6*y + 6
    return np.array([f_x, f_y])

# Gradient Descent
i = 0
while True:
    i += 1
    point = point - step_size * g(point)
    point_storage.append(point)

    print("Point History : {}".format(point_storage))
    print("Next Point : {}\n".format(point))
    
    compare_value = np.linalg.norm(point_storage[i] - point_storage[i-1])
    
    if compare_value < threshold:
        print('First two Norm : {}'.format(np.linalg.norm(point_storage[1] - point_storage[0])))
        print("Convergence Norm : {}".format(compare_value))
        print("Convergence at {} iterations".format(i))
        break     
        
# plotting 3d graph
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')
title = ax.set_title('objective_function')

x = np.arange(-2.5, 2.5, 0.1)
y = np.arange(-2.5, 2.5, 0.1)

X1, X2 = np.meshgrid(x, y)
graph = ax.plot_surface(X1, X2, obj_func((X1, X2)),cmap=cm.coolwarm, alpha=.5)

point_storage_np = np.array(point_storage)
computed_x1 = point_storage_np[:, 0]
computed_x2 = point_storage_np[:, 1]
computed_z = obj_func((computed_x1, computed_x2))
graph2 = ax.scatter(computed_x1, computed_x2, computed_z, color='black', linewidth = 0.01)
plt.show()
    
'''
Prespecified threshold : 0.0003
First two Norm : 0.10049875621120898
Convergence Norm : 0.00029502827894393194
Convergence at 267 iterations
'''