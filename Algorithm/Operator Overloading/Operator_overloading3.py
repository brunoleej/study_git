class Quadrangle:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def __add__(self, second):
        return Quadrangle(self.width + second, self.height + second)

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
    # 연산자 == 
    def __eq__(self, p):
        if ((self.width == p.width) and (self.height == p.height)):
            return True
        else:
            return False

rectangle1 = Figure(1, 2)
rectangle2 = Figure(1, 2)
rectangle3 = rectangle1

print(id(rectangle1))   # 2233178497568
print(id(rectangle2))   # 2233178497664
print(id(rectangle1))   # 2233178497568

print(rectangle1 is rectangle2) # False
print(rectangle1 is rectangle3) # True

print(rectangle1 == rectangle2) # True
print(rectangle1.width == rectangle2.width) # True