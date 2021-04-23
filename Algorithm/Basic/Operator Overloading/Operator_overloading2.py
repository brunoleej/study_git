# 객체 주소 확인하기와 is, == 연산자 이해하기
    # id(객체명) : 객체가 가리키는 실제 주소값
    # is 와 == 연산자 차이
        # is : 가리키는 객체 자체가 같은 경우 True
        # == : 가리키는 값들이 같은 경우 True

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
        
class Figure(Quadrangle):
    pass

rectangle1 = Quadrangle(1, 2)
rectangle2 = Quadrangle(1, 2)
rectangle3 = rectangle1

print(id(rectangle1))   # 2227581592096
print(id(rectangle2))   # 2227581592192

print(rectangle1 is rectangle2) # False
print(rectangle1 is rectangle3) # True

print(rectangle1 == rectangle2) # False
print(rectangle1.width == rectangle2.width) # True