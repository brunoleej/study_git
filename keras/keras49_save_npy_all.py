# boston, diabetes, cancer, iris, wine
# mnist, fashion, cifar10, cifar100

import numpy as np
from sklearn.datasets import load_boston,load_diabetes,load_breast_cancer,load_wine
from tensorflow.keras.datasets import mnist,fashion_mnist,cifar10,cifar100

# boston
boston = load_boston()
boston_data = boston.data
boston_target = boston.target

# np.save('./data/boston_data.npy',arr = boston_data)
# np.save('./data/boston_target.npy',arr = boston_target)

# # diabets
# diabets = load_diabetes()
# diabets_data = diabets.data
# diabets_target = diabets.target

# np.save('./data/diabetes_data.npy',arr = diabets_data)
# np.save('./data/diabetes_target.npy',arr = diabets_target)

# # breast_cancer
# breast_cancer = load_breast_cancer()
# cancer_data = breast_cancer.data
# cancer_target = breast_cancer.target

# np.save('./data/cancer_data.npy',arr=cancer_data)
# np.save('./data/cancer_target.npy',arr=cancer_target)

# # wine
# wine = load_wine()
# wine_data = wine.data
# wine_target = wine.target


# np.save('./data/wine_data',arr = wine_data)
# np.save('./data/wine_target',arr = wine_target)
# np.save('./data/wine.data',arr = wine_data)
# np.save('./data/wine.target',arr = wine_target)



# np.save('./data/wine.data.npy',arr = wine_data)
# np.save('./data/wine.target.npy',arr = wine_target)

# # 2~5까지 save파일을 만드시오!!!

# # 6. mnist
# (x_train,y_train),(x_test,y_test) = mnist.load_data()
# np.save('./data/mnist_x_train.npy',arr=x_train)
# np.save('./data/mnist_y_train.npy',arr=y_train)
# np.save('./data/mnist_x_test.npy',arr=x_test)
# np.save('./data/mnist_y_test.npy',arr=y_test)

# # fashion_mnist
# (train_images,train_labels),(test_images,test_labels) = fashion_mnist.load_data()
# np.save('./data/fashion_mnist_x_train.npy',arr=train_images)
# np.save('./data/fashion_mnist_y_train.npy',arr=train_labels)
# np.save('./data/fashion_mnist_x_test.npy',arr=test_images)
# np.save('./data/fashion_mnist_y_test.npy',arr=test_labels)

# # cifar10
# (c10_x_train,c10_y_train),(c10_x_test,c10_y_test) = cifar10.load_data()
# np.save('./data/c10_x_train.npy',arr = c10_x_train)
# np.save('./data/c10_y_train.npy',arr = c10_y_train)
# np.save('./data/c10_x_test.npy',arr = c10_x_test)
# np.save('./data/c10_y_test.npy',arr = c10_y_test)

# # cifar
# (c100_x_train,c100_y_train),(c100_x_test,c100_y_test) = cifar100.load_data()

# np.save('./data/c100_x_train.npy',arr = c100_x_train)
# np.save('./data/c100_y_train.npy',arr = c100_y_train)
# np.save('./data/c100_x_test.npy',arr = c100_x_test)
# np.save('./data/c100_y_test.npy',arr = c100_y_test)