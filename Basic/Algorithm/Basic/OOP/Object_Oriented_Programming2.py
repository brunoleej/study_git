'''
# 소멸자 : __del__(self)
    # method 이므로 첫 번째 인자는 self로 설정
    # 클래스 소멸시 호출

class Quadrangle:
    def __init__(self, width: int, height: int, color : str):
        self.width = width
        self.height = height
        self.color = color
    
    def __del__(self):
        print("Quadrangle object is deleted")

square = Quadrangle(5, 5, 'black')
# Quadrangle object is deleted

del square  # Quadrangle object is deleted
'''

import math

class Figure:
    def __init__(self, length : int, name : str):
        self.length = length
        self.name = name
    
    def get_area(self):
        return (math.sqrt(3) / 2) * self.length**2 
        # (math.sqrt(3) / 2) * math.pow(self.length, 2)도 가능함
    
    def get_name(self):
        return self.name

    def __del__(self):
        print('object is deleted')
    
square = Figure(10, 'dave')
print(square.get_area(), square.get_name()) 
# 86.60254037844386 dave
# object is deleted