from sklearn.svm import LinearSVC
import numpy as np
from sklearn.metrics import accuracy_score

x_data = [[0,0],[1,0],[0,1],[1,1]]
y_data = [0,0,0,1]

# model
model = LinearSVC()

# fitting
model.fit(x_data,y_data)

# evaluate
result = model.score(x_data,y_data)
print('model.score : {}'.format(result))

# prediction
y_pred = model.predict(x_data)
print('{0}의 예측결과:{1}'.format(x_data,y_pred))

acc = accuracy_score(y_data,y_pred)
print('정확도는 : {}'.format(acc))