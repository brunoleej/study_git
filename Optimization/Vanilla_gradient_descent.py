# import module
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# Hyperparameter
threshold = 0.0003   # convergence threshold
step_size = 0.01     # step-size factor
point_storage = []   # point history

# initial point
point = np.array([-2.0, 2.0])

# decision variable
x1 = point[0]
x2 = point[1]

'''
# partial-derivate function
f_x1 = 6*x1 + 4*x2 + 2
f_x2 = 4*x1 + 6*x2
g = np.array([f_x1, f_x2])
'''

def obj_func(x1, x2):
    f_x = 1+2*x1+3*(x1**2+x2**2) + 3*x1*x2
    return f_x

def g(x):
    x1, x2 = x
    f_x1 = 6*x1 + 4*x2 + 2
    f_x2 = 4*x1 + 6*x2
    return np.array([f_x1, f_x2 ])


# Gradient Descent
i = 0
point = np.array([-2.0, 2.0])
x1, x2 = point
point_storage = []
point_storage.append(point)

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

x1 = np.arange(-2.5, 2.5, 0.1)
x2 = np.arange(-2.5, 2.5, 0.1)

X1, X2 = np.meshgrid(x1, x2)
graph = ax.plot_surface(X1, X2, obj_func(X1, X2),cmap=cm.coolwarm, alpha=.5)

point_storage_np = np.array(point_storage)
computed_x1 = point_storage_np[:, 0]
computed_x2 = point_storage_np[:, 1]
computed_z = obj_func(computed_x1, computed_x2)
graph2 = ax.scatter(computed_x1, computed_x2, computed_z, color='black', linewidth = 0.01)
plt.show()
    
'''
First two points of norm : 0.01341640786499875
Convergence point : 0.004424836657664756
Convergence at 57 iterations
'''
