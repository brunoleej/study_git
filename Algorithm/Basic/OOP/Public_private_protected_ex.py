# 원 클래스 생성하기
    # attribute : 원 반지름, 원 이름
    # method
        # 1. 원 이름 리턴 메소드
        # 2. 원 넓이 리턴 메소드

    # 참고(원 넓이 식) : 3.14 X 원 반지름 **2(원 반지름의 제곱)
        # 3. 원 길이 리턴 메소드
            # 참고(원 길이 식) : 2 X 3.14 X 원 반지름
            # 생성자에서만 attribute 값 설정 가능
            # attribute는 private으로 설정
            
'''
class Circle:
    def __init__(self, radius : float, name : str):
        self.__radius = radius
        self.__name = name
    
    def get_name(self):
        return self.__name

    def get_area(self):
        return 3.14 * self.__radius**2
    
circle = Circle(3, 'dave')
print(circle.get_name(), circle.get_area())    # dave 28.26
print()    # 
'''

class Circle:
    def __init__(self, radius : float, name : str):
        self.__radius = radius
        self.__name = name
    
    def get_name(self):
        return self.__name

    def get_area(self):
        return 3.14 * self.__radius**2
    
    def get_length(self):
        return 2 * 3.14 * self.__radius

circle = Circle(3, 'dave')
print(circle.get_name(), circle.get_area(), circle.get_length())    # dave 28.26 18.84
