# private, protected, public
# 정보 은닉(Information Hiding) 방식
    # class의 attribute, method에 대해 접근을 제어할 수 있는 기능

# private -> protected -> public
    # private : private로 선언된 attribute, method는 해당 클래스에서만 접근 가능
    # protected : protected로 선언된 attribute, method는 해당 클래스 또는 해당 클래스를 상속받은 클래스에서만 접근 가능
    # public : public으로 선언된 attribute, method는 어떤 클래스라도 접근 가능

class Quadrangle:
    def __init__(self, width : int, height : int, color : str):
        self.width = width
        self.height = height
        self.color = color

    def get_area(self):
        return self.width * self.height
    
    def set_area(self, width, height):
        self.width = width
        self.height = height

square = Quadrangle(5, 5, 'black')
print(square.get_area())    # 25
print(square.width) # 5
square.width = 10
print(square.get_area())    # 50

print(dir(square))
'''
['__class__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__',
 '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', '__init__',
 '__init_subclass__', '__le__', '__lt__', '__module__', '__ne__', '__new__', 
 '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', 
 '__str__', '__subclasshook__', '__weakref__', 'color', 'get_area', 'height', 
 'set_area', 'width']
'''