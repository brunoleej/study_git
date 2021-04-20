# 리스트 변수를 활용해서 해쉬 테이블 구현하기
    # 1. 해쉬 함수 : key % 8
    # 2. 해쉬 키 생성 : hash(data)

hash_table = list([0 for i in range(8)])

def get_key(data):
    return hash(data)

def hash_function(key):
    return key % 8

def save_data(data, value):
    hash_address = hash_function(get_key(data))
    hash_table[hash_address] = value

def read_data(data):
    hash_address = hash_function(get_key(data))
    return hash_table[hash_address]

save_data('Dave','0102030200')
save_data('Andy','010332322200')
print(read_data('Dave'))    # 0102030200
print(hash_table)   # [0, '0102030200', 0, 0, 0, '010332322200', 0, 0]

