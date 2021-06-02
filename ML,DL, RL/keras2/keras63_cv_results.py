# 61_1 copy
# print model.cv_results 붙여서 완성
# DNN & ML
# from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.datasets import mnist

# Data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
from tensorflow.keras.utils import to_categorical

# Preprocessing
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000, 28*28).astype('float32')/255.
x_test = x_test.reshape(10000, 28*28).astype('float32')/255.

# Modeling
def build_model(drop=0.5, optimizer='adam') :
    inputs = Input(shape=(28*28,), name='input')
    x = Dense(512, activation='relu', name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(256, activation='relu', name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(128, activation='relu', name='hidden3')(x)
    x = Dropout(drop)(x)
    outputs = Dense(10, activation='softmax', name='outputs')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, metrics=['acc'], loss='categorical_crossentropy')
    return model

# model2 = build_model()
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
model2 = KerasClassifier(build_fn=build_model, verbose=1)   

def create_hyperparameters() :
    batches = [16, 32, 64]
    optimizers = ['rmsprop', 'adam']
    dropout = [0.1, 0.2, 0.3]
    return {"batch_size" : batches, "optimizer" : optimizers, "drop" : dropout}

hyperparameters = create_hyperparameters()

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
# search = RandomizedSearchCV(model2, hyperparameters, cv=2)
search = GridSearchCV(model2, hyperparameters, cv=2)

grid_result = search.fit(x_train, y_train, verbose=1)

print("best_params : ", search.best_params_)         
# best_params :  {'optimizer': 'adam', 'drop': 0.1, 'batch_size': 30}
print("best_estimator : ", search.best_estimator_)   
# best_estimator :  <tensorflow.python.keras.wrappers.scikit_learn.KerasClassifier object at 0x000001C0872F5DF0>
print("best_score : ", search.best_score_)           
# best_score :  0.9562999904155731 

acc = search.score(x_test, y_test)
print("Score : ", acc)
# Score :  0.9675999879837036
# keras DNN acc :  0.9828000068664551

cv_result = search.cv_results_
print("cv_result :", cv_result)

import pandas as pd
result = pd.DataFrame(cv_result)    # GridSearchCV 만 가능
print(result)
'''
    mean_fit_time  std_fit_time  mean_score_time  std_score_time param_batch_size  ... split0_test_score split1_test_score mean_test_score  std_test_score  rank_test_score
0        3.760519      0.289724         1.854113        0.100456               16  ...          0.954167          0.944067        0.949117        0.005050               12
1        3.060011      0.099090         1.944005        0.062535               16  ...          0.952600          0.945633        0.949117        0.003483               11
2        3.737402      0.073509         1.854222        0.115042               16  ...          0.948233          0.950367        0.949300        0.001067               10
3        2.861855      0.016229         1.843626        0.059288               16  ...          0.953033          0.954100        0.953567        0.000533                2
4        3.475805      0.078134         1.844951        0.102781               16  ...          0.950667          0.947067        0.948867        0.001800               13
5        2.956033      0.200059         1.832552        0.073339               16  ...          0.947733          0.948933        0.948333        0.000600               14
6        2.273422      0.070371         1.114887        0.012540               32  ...          0.955833          0.943733        0.949783        0.006050                9
7        1.995903      0.035948         1.165224        0.074295               32  ...          0.950400          0.957967        0.954183        0.003783                1
8        2.250424      0.008901         1.122972        0.004226               32  ...          0.949933          0.949767        0.949850        0.000083                8
9        2.031466      0.077758         1.132447        0.004981               32  ...          0.952267          0.951400        0.951833        0.000433                6
10       2.320675      0.036544         1.132502        0.004429               32  ...          0.939000          0.942000        0.940500        0.001500               18
11       2.076113      0.095516         1.137529        0.002446               32  ...          0.953800          0.950700        0.952250        0.001550                4
12       1.395981      0.001739         0.623500        0.000260               64  ...          0.943200          0.950567        0.946883        0.003683               16
13       1.309693      0.083195         0.638445        0.020471               64  ...          0.949433          0.954267        0.951850        0.002417                5
14       1.470706      0.068252         0.642324        0.004725               64  ...          0.953200          0.950300        0.951750        0.001450                7
15       1.254621      0.013361         0.626285        0.008641               64  ...          0.954000          0.951833        0.952917        0.001083                3
16       1.489383      0.083829         0.644854        0.029684               64  ...          0.936500          0.950600        0.943550        0.007050               17
17       1.316956      0.073738         0.664194        0.010238               64  ...          0.946633          0.947333        0.946983        0.000350               15
'''