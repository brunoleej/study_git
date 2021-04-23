# 정삼각형 클래스를 만들고, 너비 출력하기

import math

class Quadrangle:
    def __init__(self, length : int):
        self.length = length

    def get_area(self):
        return (math.sqrt(3) / 2) * self.length**2
        # (math.sqrt(3) / 2) * math.pow(self.length, 2)도 가능함

square = Quadrangle(10)
print(square.get_area())    # 86.60254037844386