import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist

# preprocessing
(train_images,train_labels),(test_images,test_labels) = fashion_mnist.load_data()
print(train_images[0].shape)    # (28, 28)

plt.imshow(train_images[0],'gray')
plt.show()
