# python Comprehension
    # 다른 Sequence로 부터 새로운 Sequence (Iterable Object)를 만들 수 있는 기능

# List Comprehension
# 사용법
# [출력표현식 for 요소 in 입력 Sequence [if 조건식]]

# 입력 Sequence는 Iteration이 가능한 데이터 Sequence 혹은 컬렉션
# [if 조건식]에서 []은 리스트 괄호가 아니라, 옵션이라는 뜻, 즉 조건이 있을때만 넣으면 된다는 뜻임

# 예 : 종류가 다른 데이터에서 정수 리스트만 가져오기
dataset = [4, True, 'Dave', 2.1, 3]
int_data = [num for num in dataset if type(num) == int]

print(int_data) # [4, 3]
print(type(int_data))   # <class 'list'>

# 출력 표현식을 num * num 으로 변경
int_square_data = [num * num for num in dataset if type(num) == int]
print(int_square_data)  # [16, 9]
