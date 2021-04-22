# 충돌(Collision) 해결 알고리즘 (좋은 해쉬 함수 사용하기)
    # 해쉬 테이블의 가장 큰 문제는 충돌(Collision)의 경우입니다. 이 문제를 충돌(Collision)또는 해쉬 충돌(Hash Collision)이라고 부름.

# Chaining 기법
    # 개방 해슁 또는 Open Hashing 기법 중 하나 : 해쉬 테이블 저장공간 외의 공간을 활용하는 기법
    # 충돌이 일어나면, 링크드 리스트라는 자료구조를 사용해서, 링크드 리스트로 데이터를 추가로 뒤에 연결시켜서 저장하는 기법

# Hash_ex1 연습의 해쉬 테이블 코드에 Chaining 기법으로 충돌해결 코드를 추가
    # 1. 해쉬 함수 : key % 8
    # 2. 해쉬 키 생성 : hash(data)1

import Hash_ex1

