# 사용법
    # {key : Value for 요소 in 입력 Sequence [if 조건식]}
    # 입력 Sequence로부터 조건에 맞는 새로운 Set 컬렉션을 리턴
    # [if 조건식]에서 [] 은 리스트 괄호가 아니라, 옵션이라는 뜻, 즉 조건이 있을때만 넣으면 된다는 뜻임

id_name = {1 : 'Dave', 2 : 'David', 3 : 'Anthony'}
print(id_name.items())  # dict_items([(1, 'Dave'), (2, 'David'), (3, 'Anthony')])

# 아이디가 1이상인 데이터를 이름 : 아이디 형식으로 새로운 set 만들기
name_id = {val : key for key,val in id_name.items() if key > 1}
print(name_id)  # {'David': 2, 'Anthony': 3}

# 아이디를 10단위로 한번에 바꾸기
name_id = {key * 10 : val for key,val in id_name.items()}
print(name_id)  # {10: 'Dave', 20: 'David', 30: 'Anthony'}