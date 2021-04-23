# 연산자 중복 정의(Operator Overloading)
    # 객체에서 필요한 연산자를 재정의하는 것
    # 연산자 중복을 위해 미리 정의된 특별한 메소드 존재 : __로 시작 _로 끝나는 특수 함수
    # 해당 메소드들을 구현하면, 객체에 여러가지 파이썬 내장 함수나 연산자를 재정의하여 사용 가능

'''
# Quadrangle + Figure = Quadrangle(widths, heights)
class Figure:
    def __init__(self, width, height):
        self.width = width
        self.height = height
    
class Quadrangle:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def __add__(self, second):
        return Quadrangle(self.width + second.width, self.height + second.height)

rectangle1 = Quadrangle(2,3)
figure1 = Figure(3,4)
rectangle2 = rectangle1 + figure1
print(rectangle2.width) # 5
'''

class Quadrangle:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def __add__(self, second):
        return Quadrangle(self.width + second.width, self.height + second.height)

    # 연산자 곱셈
    def __mul__(self, num):
        return Quadrangle(self.width * num, self.height * num)

    # 연산자 len() - 길이
    def __len__(self):
        return self.width * 2 + self.height * 2

    # 연산자 A[index] - 리스트
    def __getitem__(self, index):
        if index == 0:
            return self.width
        elif index == 1:
            return self.height
        
    # 연산지 str() - 문자열 반환
    def __str__(self):
        return 'width : {}, height : {}'.format(self.width,self.height)
        