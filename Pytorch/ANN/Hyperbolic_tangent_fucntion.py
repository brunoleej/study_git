import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5.0,5.0, 0.1)
y = np.tanh(x)

plt.plot(x,y)
plt.plot([0,0],[1.0,-1.0],":")
plt.axhline(y = 0, color = 'orange', linestyle = '--')
plt.title('Hyperbolic Tangent Funtion')
plt.show()
