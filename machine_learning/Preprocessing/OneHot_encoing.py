import numpy as np
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

items = ['TV','냉장고','전자레인지','컴퓨터','선풍기','선풍기','믹서','믹서']

# 먼저 숫자 값으로 변환을 위해 LabelEncoder로 변환합니다.
encoder = LabelEncoder()
encoder.fit(items)
labels = encoder.transform(items)
# 2차원 데이터로 변환합니다
labels = labels.reshape(-1,1)

# OneHotEncoding
one_encoder = OneHotEncoder()
one_encoder.fit(labels)
one_labels = one_encoder.transform(labels)
print('OneHot Encoding Data')
print(one_labels.toarray())
print('OneHot Encoding Data Shape')
print(one_labels.shape)