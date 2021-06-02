# 자연어 처리
# 긍정 / 부정 텍스트를 맞춰본다.

from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np

docs = ['너무 재밌어요', '참 최고에요', '참 잘 만든 영화예요', '추천하고 싶은 영화입니다',
        '한 번 더 보고 싶네요','글쎄요','별로에요','생각보다 지루해요',
        '연기가 어색해요','재미없어요','너무 재미없다','참 재밌네요','규현이가 잘 생기긴 했어요']

# 긍정 1, 부정 0
labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1,1])  # y : (13,)

token = Tokenizer()
token.fit_on_texts(docs)
print(token.word_index)
# {'참': 1, '너무': 2, '잘': 3, '재밌어요': 4, '최고에요': 5, '만든': 6, '영화예요': 7, '추천하고': 8, '싶은': 9, '영화입니다': 10, '한': 11, '번': 12, '더': 13, '보고': 14, '싶네요': 15, '글쎄요': 16, '별로에요': 17, '
# 생각보다': 18, '지루해요': 19, '연기가': 20, '어색해요': 21, '재미없어요': 22, '재미없다': 23, '재밌네요': 24, '규현이가': 25, '생기긴': 26, '했어요': 27}

x = token.texts_to_sequences(docs)
print(x)
# [[2, 4], [1, 5], [1, 3, 6, 7], [8, 9, 10], [11, 12, 13, 14, 15], [16], [17], [18, 19], [20, 21], [22], [2, 23], [1, 24], [25, 3, 26, 27]]
# 문제점 : 문장의 길이가 다 다르다.
# 해결점 : 가장 긴 문장을 기준으로 문장의 길이를 맞춰준다. 나머지 공간은 0으로 채워준다. 
# 주의점 : 모델은 뒤로갈 수록 영향력이 크기 때문에 앞에서 0을 채운다.

from tensorflow.keras.preprocessing.sequence import pad_sequences
# pad_x = pad_sequences(x, padding='pre', maxlen=4)  # padding=pre : 앞에서부터 0을 채운다. & maxlen=4 : 최대 길이를 조정할 수 있다. >> 앞에 있는 데이터를 자른다.
# pad_x = pad_sequences(x, padding='post', maxlen=4) # padding=post : 뒤에서부터 0을 채운다. & maxlen=4 : 최대 길이를 조정할 수 있다. >> 뒤에 있는 데이터를 자른다.
pad_x = pad_sequences(x, padding='pre', maxlen=5)  

print(pad_x)
print(pad_x.shape)  # (13, 5)

print(np.unique(pad_x))
print(len(np.unique(pad_x)))
# [ 0  1  2  3  4  5  6  7  8  9 10 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27] >>> 11이 maxlen 길이 초과로 인해 잘렸다.
# [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27] 
# 28

# 원핫인코딩의 단점 : 문자의 길이가 길어질 경우, 용량이 너무 커진다. 의미가 없는 공간들이 너무 많다.
# 해결점 Embedding : 임베딩 레이어에서 수치화(백터화)해준다. >> 문자의 길이를 output_dim 길이만큼 맞춰준다.

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, LSTM, Flatten, Conv1D, BatchNormalization

model = Sequential()
# [1] Embedding
model.add(Embedding(input_dim=28, output_dim=11, input_length=5))  
# model.add(Embedding(input_dim=8, output_dim=11, input_length=5))  # input_dim 28개보다 작을 때 >>  안된다. >> 데이터의 개수를 판단해서 28개보다 작으면 통과안시킴
# model.add(Embedding(input_dim=80, output_dim=11, input_length=5)) # input_dim 28개보다 클 때   >>  된다.   >> 28개보다 더 많은 건 상관없다.
# input_dim=28 : 워드 사이즈, 총 단어의 개수를 명시해준다. >> 실제 단어사전의 개수보다 같거나 커야 한다.
# output_dim : output, 다음 레이어로 전달하는 파라미터의 개수
# input_length=5 : 단어의 자릿수를 넣어준다. 컬럼 수
'''
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
embedding (Embedding)        (None, 5, 11)             308          <== 연산의 개수 : input_dim * output_dim
=================================================================
Total params: 308
Trainable params: 308
Non-trainable params: 0
_________________________________________________________________
'''
# 3차원으로 아웃풋 나온다.

# [2] Embedding - 뒤에 파라미터를 명시하지 않아도 됨 / input_length=None
# model.add(Embedding(28, 11))  # >> Flatten 안 먹힌다.
'''
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
embedding (Embedding)        (None, None, 11)          308
=================================================================
Total params: 308
Trainable params: 308
Non-trainable params: 0
_________________________________________________________________
'''
# model.add(LSTM(32))
model.add(Conv1D(32, 2))
model.add(Flatten())    # flatten 없이도 작동하기는 함, 하지만 더 좋은 성능을 얻기 위해서 해주자.
model.add(Dense(1, activation='sigmoid'))

model.summary()
'''
* [1] Conv1D 적용
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
embedding (Embedding)        (None, 5, 11)             308
_________________________________________________________________
conv1d (Conv1D)              (None, 4, 32)             736
_________________________________________________________________
flatten (Flatten)            (None, 128)               0
_________________________________________________________________
dense (Dense)                (None, 1)                 129
=================================================================
Total params: 1,173
Trainable params: 1,173
Non-trainable params: 0
_________________________________________________________________

* [2] Flatten 적용
>> 안됨

'''

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
model.fit(pad_x, labels, epochs=100)

acc = model.evaluate(pad_x, labels)[1]
print("accuracy : ", acc)
# accuracy :  1.0

