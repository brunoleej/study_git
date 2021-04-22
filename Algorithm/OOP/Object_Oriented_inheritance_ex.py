# Object_Oriented_inheritance 파일의 코드가 실행된 후, 최종 값을 예상해보소, 왜 이와 같이 작성하였는지 이해하기(상속 개념 이해)

class Figure:
    def __init__(self, name : str, color : str):
        self.name = name
        self.color = color

class Qudrangle(Figure):
    def set_area(self, width, height):
        self.__width = width
        self.__height = height

    def get_info(self):
        print(self.name, self.color, self.__width * self.__height)

square = Qudrangle('square1', 'black')
square.set_area(5,5)
print(square.get_info())    # square1 black 25
