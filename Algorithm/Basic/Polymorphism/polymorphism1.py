# 다형성(polymorphism)
# 같은 모양의 코드가 다른 동작을 하는 것
# 키보드의 예로
    # push(keyboard): 키보드를 누른다는 동일한 코드에 대해
    # ENTER, ESC, A 등 실제 키에 따라 동작이 다른 것을 의미함
# 다형성은 코드의 양을 줄이고, 여러 객체 타입을 하나의 타입으로 관리가 가능하여 유지보수에 좋음

# Method Override (메소드 재정의)도 다형성의 한 예입니다.

# 클래스 선언
class Person:
    def __init__(self, name):
        self.name = name
    
    def work(self):
        print(self.name + " works hard")

class Student(Person):
    def work(self):
        print(self.name + " studies hard")
    
class Engineer(Person):
    def work(self):
        print(self.name + ' develops something')

# 객체 생성
student1 = Student('Dave')
developer1 = Engineer('David')
print(student1.work())
print(developer1.work())