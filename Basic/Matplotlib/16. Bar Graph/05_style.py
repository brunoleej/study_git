import matplotlib.pyplot as plt
import numpy as np

x = np.arange(3)
years = ['2018', '2019', '2020']
values = [100, 400, 900]

plt.bar(x, values, align='edge', edgecolor='lightgray',
        linewidth=5, tick_label=years)

plt.show()