# 아래 코드는 최상위 클래스 메소드를 두 번 호출하게 되는 문제점이 있음 (심화)

'''
# 클래스 선언
class Person:
    def __init__(self):
        print('I am a person')
    
class Student(Person):
    def __init__(self):
        Person.__init__(self)
        print('I am a student')

class Worker(Person):
    def __init__(self):
        Person.__init__(self)
        print('I am a worker')

# 다중 상속
class PartTimer(Student, Worker):
    def __init__(self):
        Student.__init__(self)
        Worker.__init__(self)
        print('I am a part-timer and student')

parttimer1 = PartTimer()


# I an a person
# I an a student
# I an a person
# I am a worker
# I an a part-timer and student
'''

# super() 내장함수를 사용하면 위의 문제를 해결할 수 있음
# 클래스 선언
class Person:
    def __init__(self):
        print('I am a person')
    
class Student(Person):
    def __init__(self):
        super().__init__()
        print('I am a student')

class Worker(Person):
    def __init__(self):
        super().__init__()
        print('I am a worker')

# 다중 상속
class PartTimer(Student, Worker):
    def __init__(self):
        super().__init__()
        print('I am a part-timer and student')
    
parttimer1 = PartTimer()

'''
I an a person
I am a worker
I an a student
I an a part-timer and student
'''

# 다양한 상속구조에서 메소드 이름을 찾는 순서는 '__mro__'에 튜플로 정의되어 있음
    # MRO : Method Resolution Order 의 약자

print(PartTimer.__mro__)   

'''
(<class '__main__.PartTimer'>, <class '__main__.Student'>, 
<class '__main__.Worker'>, <class '__main__.Person'>, <class 'object'>)
'''