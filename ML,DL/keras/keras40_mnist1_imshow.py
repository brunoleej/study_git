import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from keras.datasets import mnist

(x_train,y_train),(x_test,y_test) = mnist.load_data()
print(x_train.shape,y_train.shape)  # (60000, 28, 28) (60000,)
print(x_test.shape,y_test.shape)    # (10000, 28, 28) (10000,)    
print(np.min(x_train),np.max(x_test))   # 0 255

print(x_train[0])
print('y_train[0 : ',y_train[0])
print(x_train[0].shape) # (28, 28)

plt.imshow(x_train[0],'jet')
plt.colorbar()
plt.show()