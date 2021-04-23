# Set Comprehension
# 사용법
    # {출력표현식 for 요소 in 입력Sequence [if 조건식]}
    # 입력 Sequence로부터 조건에 맞는 새로운 Set 컬렉션을 리턴
    # [if 조건식]에서 []은 리스트 괄호가 아니라, 옵션이라는 뜻, 즉 조건이 있을때만 넣으면 된다는 뜻임

int_data = [1,1,2,3,3,4]

# 예 : num * num 의 set 컬렉션 만들기
square_data_set = {num * num for num in int_data}
print(square_data_set)  # {16, 1, 4, 9}

# 예 : num * num 의 set 컬렉션 만들기 (조건 붙여보기)
square_data_set2 = {num * num for num in int_data if num > 3}
print(square_data_set2) # {16}