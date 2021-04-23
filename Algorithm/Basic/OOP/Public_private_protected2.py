# protected
    # 파이썬에서는 해당 속성 앞에 _(single underscore)를 붙여서 표시만 함
    # 실제 제약되지는 않고 일종의 경고 표시로 사용됨

class Quadrangle:
    def __init__(self, width : int, height: int, color : str):
        self._width = width
        self._height = height 
        self._color = color

    def get_area(self):
        return self._width * self._height
    
    def _set_area(self, width, height):
        self._width = width
        self._height = height

square = Quadrangle(5,5,'black')
print(square.get_area())    # 25
print(square._width)    # 5
square._width = 10
print(square.get_area())    # 50
square._set_area(3, 3)  
print(square.get_area())    # 9