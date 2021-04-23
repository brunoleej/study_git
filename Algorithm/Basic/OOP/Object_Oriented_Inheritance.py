# Class Inheritance (상속)
# 추상화(abstraction) : 여러 클래스에 중복되는 속성, 메소드를 하나의 기본 클래스로 작성하는 작업
# 상속(inheritance) : 기본 클래스의 공통 기능을 물려받고, 다른 부분만 추가 또는 변경하는 것
    # 이 떄 기본 클래스는 부모 클래스(또는 상위 클래스), Parent, Super, Base class라고 부름
    # 기본 클래스 기능을 물려받는 클래스는 자식 클래스(또는 하위 클래스), Child, Sub, Derived class 라고 부름

# 코드 재사용이 가능, 공통 기능의 경우 기본 클래스 코드만 수정하면 된다는 장점
# 부모 클래스가 둘 이상인 경우 다중 상속이라고 부름

# 공통점과 차이점 찾아보기
    # 사각형 : 사각형 이름, 사각형 색, 사각형 너미 / 높이, 사각형 넓이
    # 삼각형 : 삼각형 이름, 삼각형 색, 삼각형 한 변 길이, 삼각형 넓이
    # 원 : 원 이름, 원 색, 원 반지름, 원 넓이

# 부모 클래스를 자식 클래스에 인자로 넣으면 상속이 됨
# 다음 코드는 __init__(self, name, color) 메서드가 상속되고
# self.name과 self.color 도 __init__ 실행 시 생성됨

class Figure:
    def __init__(self, name : str, color : str):
        self.name = name
        self.color = color

class Qudrangle(Figure):
    def set_area(self, width, height):
        self.__width = width
        self.__height = height

    def get_info(self):
        print(self.name, self.color, self.__width * self.__height)

square = Qudrangle('dave', 'blue')
square.set_area(5,5)    
print(square.get_info())    # dave blue 25


# 상속 관계인 클래스 확인하기
    # 내장함수 issubclass(자식 클래스, 부모 클래스) 사용하기
print(issubclass(Qudrangle, Figure))    #  True


# 클래스와 객체간의 관계 확인하기
    # 내장함수 isinstance(객체, 클래스) 사용하기
figure1 = Figure('figure1', 'black')
square = Qudrangle('square', 'red')

print(isinstance(figure1,Figure))   # True
print(isinstance(square,Figure))    # True
print(isinstance(figure1,Qudrangle))    # False
print(isinstance(square,Qudrangle))    # True
