import matplotlib.pyplot as plt
from keras.datasets import cifar100

(x_train,y_train),(x_test,y_test) = cifar100.load_data()

plt.imshow(x_train[0])
plt.show()