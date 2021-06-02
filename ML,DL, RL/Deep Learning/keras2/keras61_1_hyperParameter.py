# DNN과 머신러닝을 엮는다.
# from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

#1. DATA / Preprocessing
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000, 28*28).astype('float32')/255.
x_test = x_test.reshape(10000, 28*28).astype('float32')/255.

#2. Modeling
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

##############################################################
# 함수형 모델을 KerasClassifier로 wrapping해야 한다.
# keras 모델을 sklearn작업에 사용할 수 있다.
# build_fn: 호출가능한 함수 혹은 클레스 인스턴스
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
model2 = KerasClassifier(build_fn=build_model, verbose=1)   
##############################################################

def create_hyperparameters() :
    batches = [10, 20, 30, 40, 50]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropout = [0.1, 0.2, 0.3]
    return {"batch_size" : batches, "optimizer" : optimizers, "drop" : dropout}

hyperparameters = create_hyperparameters()

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
search = RandomizedSearchCV(model2, hyperparameters, cv=2)
# search = GridSearchCV(model2, hyperparameters, cv=2)

search.fit(x_train, y_train, verbose=1)
# keras 모델을 그냥 사용하면 타입에러 뜬다.
# TypeError: If no scoring is specified, the estimator passed should have a 'score' method. The estimator <tensorflow.python.keras.engine.functional.Functional object at 0x0000015AA3BC7E20> does not.
# >> KerasClassifier 넣어줘야 함

print("best_params : ", search.best_params_)         
# 내가 선택한 파라미터 중에서 좋은 것
# best_params :  {'optimizer': 'adam', 'drop': 0.1, 'batch_size': 30}
print("best_estimator : ", search.best_estimator_)   
# 전체 파라미터에서 좋은 것 >> sklearn에서는 인식하지 못한다.
# best_estimator :  <tensorflow.python.keras.wrappers.scikit_learn.KerasClassifier object at 0x000001C0872F5DF0>
print("best_score : ", search.best_score_)           
# best_score :  0.9562999904155731 

acc = search.score(x_test, y_test)
print("Score : ", acc)
# Score :  0.9675999879837036


# keras DNN acc :  0.9828000068664551