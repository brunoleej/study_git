# weight 저장
# model.save() 사용
# pickle 사용후 modelcheckpoint랑 비교
import numpy as np
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.datasets import mnist
from sklearn.metrics import r2_score

# Data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocessing
from tensorflow.keras.utils import to_categorical
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
    batches = [8, 32]
    optimizers = ['rmsprop', 'adam']
    dropout = [0.2, 0.3]
    return {"batch_size" : batches, "optimizer" : optimizers, "drop" : dropout}

hyperparameters = create_hyperparameters()

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
search = RandomizedSearchCV(model2, hyperparameters, cv=2)
# search = GridSearchCV(model2, hyperparameters, cv=2)

search.fit(x_train, y_train, verbose=1)
# search.save('../data/h5/k64_save1.h5')

print("best_params : ", search.best_params_)         
print("best_estimator : ", search.best_estimator_)   
print("best_score : ", search.best_score_)           

acc = search.score(x_test, y_test)
print("Score : ", acc)

# model.save
search.best_estimator_.model.save('../data/h5/k64_save.h5')

# y_pred = search.predict(x_test)
# r2 = r2_score(y_test, y_pred)
# print("r2 : ", r2)

# print("========model load========")
# model3 = load_model('../data/h5/k64_save.h5')
# print("best_score : ", model3.best_score_)           

import pickle   # Fail
pickle.dump(search, open('../data/h5/k64.pickle.data', 'wb')) # wb : write
print("== save complete ==")

print("========pickle load========")
# model3 = pickle.load(open('../data/h5/k64.pickle.data', 'rb'))
# print("== load complete ==")
# r2_2 = model3.score(x_test, y_test)
# print("r2_2 : ", r2_2)
