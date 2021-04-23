class Person:
    def __init__(self, name):
        self.name = name
    
class Student(Person):
    def study(self):
        print(self.name + " studies hard")

class Employee(Student):
    def work(self):
        print(self.name + " works hard")

# 객체 생성
student1 = Student('Dave')
employee1 = Employee('David')

# 객체 실행
print(student1.study()) # Dave studies hard
print(employee1.work()) # David works hard

