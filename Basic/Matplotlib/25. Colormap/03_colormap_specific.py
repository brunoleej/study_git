import matplotlib.pyplot as plt
from matplotlib import cm

cmaps = plt.colormaps()
for cm in cmaps:
    print(cm)