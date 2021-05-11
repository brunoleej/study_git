# 아래 코드를 실행시키면서, 출력 값이 왜 이렇게 나왔는지 이해하기(연산자 오버로딩 이해)

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

rectangle1 = Quadrangle(2,3)    
rectangle3 = rectangle1 + 4
print(rectangle3.width) # 6
print(rectangle3.width, rectangle3.height)  # 6 7

rectangle4 = rectangle1 * 3
print(str(rectangle1))  # width : 2, height : 3
print(str(rectangle4))  # width : 6, height : 9
print(len(rectangle1))  # 10