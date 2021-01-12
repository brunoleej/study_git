# summary
# 상속 : 다른 클래스를 받아서 기능을 추가해서 새로운 클래스를 만드는 방법
# super : 부모 클래스에서 특정 함수의 코드를 가져오는 방법
# getter,setter : 클래스로 만들어진 객체에 변수값을 수정하거나 출력할 때 특정 함수를 통해서 수정하고 출력하는 방법
# non public(private) ; mangling(__)`_(클래스명)`이 붙은 변수로 객체를 생성할 때 변경이 되어서 생성
# is a / has a: 클래스를 설계하는 방법
# magic(special) method
    # 비교 : `__eq__`(==),`__ne__`(!=),`__lt__`(<),`__gt__`(>),`__le__`(<=),`__ge__`(>=)
    # 연산 : `__add__`(+),`__sub__`(-),`__mul__`(*),`__truediv__`(/),`__floordiv__`(//),`__mod__`(%),`__pow__`(**)
    # 그 외 : `__repr__`,`__str__`

# Integer 객체 만들
# a = 1
class Integer:
    def __init__(self,number):
        self.number = number

    def __add__(self,obj):
        return self.number + obj.number

    def __repr__(self):
        return self.number

    def __str__(self):
        return str(self.number)

num1 = Integer(1)
num2 = Integer(2)
print(num1 + num2)  # 3
print(num1) # 1
print(num2) # 2

# Class 문제
# 계좌 Class : 변수 자산(asset),이자율(interest)
# 함수 : 인출(draw), 입금(insert), 이자추가(add_interest)
# 인출 시 자산 이상의 돈을 인출할 수 없습니다.

class Account:
    def __init__(self,asset,interest):
        self.asset = asset
        self.interest = interest
    
    def draw(self,money):
        self.money = money
        if self.asset < self.money:
            print('draw denied')
        else:
            return self.asset - self.money
    
    def insert(self):
        return self.money + self.asset
    
    def add_interest(self):
        
