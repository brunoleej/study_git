# CNN과 머신러닝을 엮는다.
# from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
# 노드의 개수를 파라미터 튜닝에 넣는다.

import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D, MaxPool2D, Flatten
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

#1. DATA / Preprocessing
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255.
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255.

#2. Modeling
def build_model(drop=0.5, optimizer='adam', node=16, activation='relu', kernel_size=2, lr = 0.01) :
    inputs = Input(shape=(28,28,1), name='input')
    x = Conv2D(512, kernel_size , activation=activation ,padding='same' , name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Conv2D(256, kernel_size , activation=activation ,padding='same' , name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Conv2D(128, kernel_size , activation=activation ,padding='same' , name='hidden3')(x)
    x = Dropout(drop)(x)
    x = Flatten()(x)
    x = Dense(node, activation='relu', name='dense1')(x)
    x = Dense(node, activation='relu', name='dense2')(x)
    outputs = Dense(10, activation='softmax', name='outputs')(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, metrics=['acc'], loss='categorical_crossentropy')
    return model

model2 = build_model()

def create_hyperparameters() :
    batches = [16, 32, 64]
    optimizers = ['rmsprop', 'adam', 'adadelta', 'sgd']
    dropout = [0.1, 0.2, 0.3]
    kernel_size = [2, 3]
    node = [128, 64, 32]
    activation =['relu','elu','prelu', 'softmax']
    lr = [0.1, 0.05, 0.01, 0.005, 0.001]
    return {"batch_size" : batches, "optimizer" : optimizers, "drop" : dropout, \
        "activation" : activation, "node" : node, "kernel_size" : kernel_size, "lr" : lr}

hyperparameters = create_hyperparameters()

# 함수형 모델을 KerasClassifier로 wrapping
# keras 모델을 sklearn작업에 사용할 수 있다.
# build_fn: 호출가능한 함수 혹은 클레스 인스턴스
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
model2 = KerasClassifier(build_fn=build_model, verbose=1)   

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
search = RandomizedSearchCV(model2, hyperparameters, cv=2)
# search = GridSearchCV(model2, hyperparameters, cv=2)

search.fit(x_train, y_train, verbose=1)
# keras 모델을 그냥 사용하면 타입에러 뜬다.
# TypeError: If no scoring is specified, the estimator passed should have a 'score' method. The estimator <tensorflow.python.keras.engine.functional.Functional object at 0x0000015AA3BC7E20> does not.
# >> KerasClassifier 넣어줘야 함

print("best_params : ", search.best_params_)         
# 내가 선택한 파라미터 중에서 좋은 것
# best_params :  {'optimizer': 'adam', 'node': 128, 'lr': 0.001, 'kernel_size': 3, 'drop': 0.2, 'batch_size': 16, 'activation': 'elu'}
print("best_estimator : ", search.best_estimator_)   
# 전체 파라미터에서 좋은 것 >> sklearn에서는 인식하지 못한다.
# best_estimator :  <tensorflow.python.keras.wrappers.scikit_learn.KerasClassifier object at 0x000001C0872F5DF0>
print("best_score : ", search.best_score_)           
# best_score :  0.9355666637420654

acc = search.score(x_test, y_test)
print("Score : ", acc)
# Score :  0.8720999956130981


# keras CNN acc :  0.9889000058174133