#   과제
#   1. 계좌 관리 class 작성하기
        # attribute : 계좌 초기 금액을 속성으로 하나 설정
        # 생성자에서 초기 금액은 0으로 설정
        # 속성은 private으로 설정
        # method : 인출, 저축, 잔액 확인 세가지 method 구현, 각각 현재 계좌 금액 리턴
        # 각 method도 private으로 설정

class Account:
    def __init__(self, account : int, money : int):
        self.__account = account
        self.__money = money 

    def out_money(self):
        self.__account -= self.__money
        return self.__account

    def in_money(self):
        self.__account += self.__money
        return self.__account

    def left_money(self):
        return self.__account

account1 = Account(3000,300)

print(account1.out_money())