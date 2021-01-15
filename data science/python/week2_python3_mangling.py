# non public
# mangling 이라는 방법으로 다이렉트로 객체의 변수에 접근하지 못하게 하는 방법

class Calc:
    def __init__(self,n1,n2):
        self.n1 = n1
        self.__n2 = n2  # mangling

    def getter(self):
        return self.__n2

    # n2에 0이 들어가지 않도록 함
    def setter(self,n2):
        n2 = 1 if n2 == 0 else num2
        self.__n2 = n2
    
    def __disp(self):
        print(self.n1,self.__n2)

    def div(self):
        self.__disp()
        return self.n1 / self.__n2
    
    number2 = property(getter,setter)

calc = Calc(1,2)

print(calc.div())   # 0.5

print(calc.number2) # getter function 실행 : 2

calc.number2 = 0
print(calc.number2) # 1

# print(calc.__n2)    #  에러가 걸림
print(calc._Calc__n2)   # 1
print(calc.div()) # 1.0

# 함수도 mangling이 가능함