# Hash Table : 키(Key)에 데이터(Value)를 저장하는 데이터 구조
    # Key를 통해 바로 데이터를 받아올 수 있으므로, 속도가 획기적으로 빨라짐
    # 파이썬 딕셔너리(Dictionary) 타입이 해쉬 테이블의 예 : Key를 가지고 바로 데이터(Value)를 꺼냄
    # 보통 배열로 미리 Hash Table 사이즈만큼 생성 후에 사용 (공간과 탐색 시간을 맞바꾸는 기법)
    # Python에서는 해쉬를 별도 구현할 필요 없음 --> 딕셔너리 타입을 이용하면 됨

# 용어
    # 해쉬(Hash) : 임의 값을 고정 길이로 변환하는 것 --> ex) SHA-256
    # 해쉬 테이블(Hash Table) : 키 값의 연산에 의해 직접 접근이 가능한 데이터 위치를 찾을 수 있는 함수
    # 해쉬 값(Hash Value) 또는 해쉬 주소(Hash Address) : Key를 해싱 함수로 연산해서, 해쉬 값을 알아내고, 이를 기반으로 해쉬 테이블에서 해당 Key에 대한 데이터 위치를 일관성 있게 찾을 수 있음
    # 슬롯(slot) : 한 개의 데이터를 저장할 수 있는 공간
    # 저장할 데이터에 대해 Key를 추출할 수 있는 별도 함수도 존재할 수 있음

# 해쉬 예
    # Hash Table 만들기
hash_table = list([i for i in range(10)])
print(hash_table)   # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# 간단한 해쉬 함수 생성
    # 다양한 해쉬 함수 고안 기법이 있으며, 가장 간단한 방법이 Division 법 (나누기를 통한 나머지 값을 사용하는 기법)
def hash_func(key):
    return key % 5

# 해쉬 테이블에 저장
    # 데이터에 따라 필요 시 key 생성 방법 정의가 필요함
data1 = 'Andy'
data2 = 'Dave'
data3 = 'Trump'

# 문자열의 인덱스의 ASCII값 반환 --> data1[0] : 문자열 처음 단어의 ASCII값 출력
print(ord(data1[0]),ord(data2[0]),ord(data3[0]))    # 65 68 84
print(hash_func(ord(data1[0])), hash_func(ord(data2[0])), hash_func(ord(data3[0]))) # 0 3 4 --> 5로 나눈 값 반환

# 해쉬 테이블에 값 저장 예시
    # data : value 와 같이 data와 value를 넣으면, 해당 data에 대한 key를 찾아서, 해당 key에 대응하는 해쉬주소에 value를 저장하는 예
def storage_data(data, value):
    key = ord(data[0])
    hash_address = hash_func(key)
    hash_table[hash_address] = value

# 해쉬 테이블에서 특정 주소의 데이터를 가져오는 함수 생성
storage_data('Andy','0105553333')
storage_data('Dave','0105553333')
storage_data('Trump','0105553333')

# 데이터 저장하고 읽어오기
def get_data(data):
    key = ord(data[0])
    hash_address = hash_func(key)
    return hash_table[hash_address]

print(get_data('Andy')) # 0105553333


# 해쉬 테이블의 장단점과 주요 용도
    # 장점
        # 데이터 저장 / 읽기 속도가 빠르다. (검색 속도가 빠르다)
        # 해쉬는 키에 대한 데이터가 있는지(중복) 확인이 쉬움
    # 단점
        # 일반적으로 저장공간이 좀더 많이 필요하다.
        # 여러 키에 해당하는 주소가 동일할 경우 충돌을 해결하기 위한 별도 자료구조가 필요함
    # 주요 용도
        # 검색이 많이 필요할 경우
        # 저장, 삭제, 읽기가 빈번한 경우
        # 캐쉬 구현시 (중복 확인이 쉽기 때문)
        