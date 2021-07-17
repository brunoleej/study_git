# 클래스 : 클래스 안에 함수, 변수 등등 많이 넣을 수 있다.

class Person :
    def __init__(self, name, age, address) : 
        # __init__ : 클래스 초기화, 가장 먼저 실행된다.
        # self : 반드시 명시되어야 함 , 클래스 자체, 자기 자신을 의미함 / 선언 할 때는 따로 명시할 필요는 없음
        # name, age, address : 입력 받야야 하는 값
        self.name = name    # 클래스 내에서 사용 가능함
        self.age = age
        self.address = address

    def greeting(self) :    # 클래스 안에 들어가는 거에는 무조건 self를 넣어줘야 한다. ***
        print("Hello, I'm {0}.".format(self.name))
        # {0} : .format() 안에 있는 것과 매칭된다.
