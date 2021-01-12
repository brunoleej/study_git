# name, age의 변수를 포함한 Person 클래스를 만들어주세요.

class Person:
    def __init__(self,name='Liam',age='21'):
        self.name = 'Liam'
        self.age = '21'

person = Person('Liam', '21')
print(person.name, person.age)