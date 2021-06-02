# 61 pipeline 추가
# DNN & ML
# from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.datasets import mnist
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold
from tensorflow.keras.utils import to_categorical

# Data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocessing
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000, 28*28).astype('float32')
x_test = x_test.reshape(10000, 28*28).astype('float32')

# Modelig
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

def create_hyperparameters() :
    batches = [8, 32]
    optimizers = ['rmsprop', 'adam']
    dropout = [0.2, 0.3]
    return {"kc__batch_size" : batches, "kc__optimizer" : optimizers, "kc__drop" : dropout}

hyperparameters = create_hyperparameters()
model2 = KerasClassifier(build_fn=build_model, verbose=1, batch_size=32, epochs=1)   
pipe = Pipeline([("scaler", MinMaxScaler()), ("kc", model2)])
kf = KFold(n_splits=2, random_state=42)
search = GridSearchCV(pipe, hyperparameters, cv=kf)

search.fit(x_train, y_train)

print("best_params : ", search.best_params_)         
print("best_score : ", search.best_score_)           
# best_params :  {'kc__batch_size': 32, 'kc__drop': 0.2, 'kc__optimizer': 'adam'}
# best_score :  0.9501833319664001

acc = search.score(x_test, y_test)
print("Score : ", acc)
# Score : 0.9692999720573425
