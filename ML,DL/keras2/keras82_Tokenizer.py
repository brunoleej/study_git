# 자연어 처리
# Tokenizer

from tensorflow.keras.preprocessing.text import Tokenizer

text = '나는 진짜 맛있는 밥을 진짜 진짜 마구 마구 먹었다.'

# 어절(띄어쓰기) 별로 자르기
token = Tokenizer()
token.fit_on_texts([text])

# word의 인덱스, 가장 빈도수가 높은 단어가 앞 인덱스로 붙여진다.
print(token.word_index)  # {'진짜': 1, '마구': 2, '나는': 3, '맛있는': 4, '밥을': 5, '먹었다': 6}

# 단어를 수치화한다.
x = token.texts_to_sequences([text])
print(x)                 # [[3, 1, 4, 5, 1, 1, 2, 2, 6]]

# 텍스트의 원핫인코딩
from tensorflow.keras.utils import to_categorical
word_size = len(token.word_index)
print(word_size)        # 6 >> 6개 컬럼으로 컬럼이 나눠진다.
x = to_categorical(x)

print(x)
print(x.shape)          # (1, 9, 7)
