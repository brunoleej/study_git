# 3. 피자 가게 관리 class 작성하기
    # attribute : 피자 종류(리스트 데이터 타입), 피자 가게 이름 속성
    # 생성자에서 각 속성을 객체 생성시 전달된 인자값으로 설정, 피자 종류는 ['슈퍼슈프림', '콤비네이션', '불고기']로 제공
    # 각 속성은 private으로 설정
    # method : 원하는 피자를 제공하는지를 알려주는 기능, YES 또는 NO 문자열을 리턴

class Pizza:
    def __init__(self, type_pizza = ["슈퍼수프림", '콤비네이션', '불고기'], name : str):
        self.__type_pizza  = type_pizza
        self.__name = name

    def is_pizza(self):
        if input(str()) in self.__type_pizza:
            return "YES"
        else:
            return "NO"

pizza = Pizza('지수의 피자가게')
print(pizza.is_pizza('슈퍼수프림'))