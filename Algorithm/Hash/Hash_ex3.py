# Linear Probing 기법
    # 폐쇄 해슁 또는 Close Hashing 기법 중 하나 : 해쉬 테이블 저장공간 안에서 충돌 문제를 해결하는 기법
    # 충돌이 일어나면, 해당 hash address의 다음 address부터 맨 처음 나오는 빈공간에 저장하는 기법
        # 저장공간 활용도를 높이기 위한 기법

# Hash_ex1의 해쉬 테이블 코드에 Linear Probling 기법으로 충돌해결 코드를 추가해보기
    # 1. 해쉬 함수 : key % 8
    # 2. 해쉬 키 생성 : hash(data)