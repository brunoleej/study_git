# 파이썬에서는 attribute, method 앞에 __(double underscore)를 붙이면 실제로 해당 이름으로 접근이 허용되지 않음
# 실은 __(double underscore)를 붙이면, 해당 이름이 _classname__ 해당 속성 또는 메소드 이름으로 변경되기 때문임

class Quadrangle:
    def __init__(self, width: int, height : int, color : str):
        self.__width = width
        self.__height = height
        self.__color = color

    def get_area(self):
        return self.__width * self.__height

    def __set_area(self, width, height):
        self.__width = width
        self.__height = height

square = Quadrangle(5, 5, 'black')
print(dir(square))
'''
['_Quadrangle__color', '_Quadrangle__height', '_Quadrangle__set_area', 
'_Quadrangle__width', '__class__', '__delattr__', '__dict__', '__dir__', 
'__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__', 
'__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__module__', 
'__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', 
'__sizeof__', '__str__', '__subclasshook__', '__weakref__', 'get_area']
'''

# print(square.__set_area(10, 10))    # Error
# print(square.__width)   # error
print(square.get_area())    # 25
