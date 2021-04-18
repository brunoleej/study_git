# 객체지향 프로그래밍(Object Oriented Programming)
# 절차지향과 객체지향 프로그래밍
# 절차지향 프로그래밍
    # 대표 언어 : 파스칼, 코볼, 포트란, C언어 등

data = 1
print(data) # 1

# 객체지향 프로그래밍
    # 객채(object) 단위로 데이터와 기능(함수)를 하나로 묶어서 쓰는 언어

# 객체지향 프로그램 작성 방법
    # 1. 클래스 설계(attribute와 method 구성)
    # 2. 설계한 클래스를 기반으로 클래스를 코드로 작성
    # 3. 클래스를 기반으로 필요한 객체 생성
    # 4. 해당 객체의 attribute와 method를 조작하여 프로그램 수행


# class 선언하기
# 객체 생성 전에 미리 class를 선언해야 함
class Quadrangle:
    pass

class SingleWord:
    pass

print(dir(SingleWord))
'''
['__class__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__',
 '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', '__init__',
 '__init_subclass__', '__le__', '__lt__', '__module__', '__ne__', '__new__',
 '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__',
 '__str__', '__subclasshook__', '__weakref__']
'''

# class도 변수 / 함수와 마찬가지로 유일한 이름을 지어줘야 함
# class에 attribute / method를 아직 넣지 않은 상태이므로, 클래스에 내용이 없기에
# 임의의 pass를 넣어 클래스 선언이 끝났음을 알려줌
    # pass는 아무것도 수행하지 않는 문법, 임시 코드 작성 시 주로 사용

# 객체도 변수 / 함수 / 클래스와 마찬가지로 유일한 이름을 지어줘야 함
# 함수와 마찬가지로 인수를 넣을 수 있음
square = Quadrangle()
# dave = Quadrangle()
# dave1 = Quadrangle()
# print(dir(dave))
'''
['__class__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__',
 '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', '__init__',
 '__init_subclass__', '__le__', '__lt__', '__module__', '__ne__', '__new__',
 '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__',
 '__str__', '__subclasshook__', '__weakref__']
'''

# 객체 square은 Qudrangle의 인스턴스
print(type(square)) # <class '__main__.Quadrangle'>

class Quadrangle:
    def __init__(self, width : int, height: int, color: str):
        self.width = width
        self.height = height
        self.color = color
    
    def get_area(self):
        return self.width * self.height

square1 = Quadrangle(5, 5, 'black')
square2 = Quadrangle(7, 7, 'red')


print(square1.get_area())   # 25
print(square2.get_area())   # 49
