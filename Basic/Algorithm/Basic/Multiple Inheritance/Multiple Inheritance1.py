# 다중 상속
    # 다중 상속 : 2개 이상의 클래스를 상속받는 경우
    # 상속 받은 모든 클래스의 attribute와 method를 모두 사용 가능

# 클래스 선언
class Person:
    def sleep(self):
        print('sleep')

class Student(Person):
    def study(self):
        print('Study hard')

class Worker(Person):
    def work(self):
        print('Work hard')

# 다중 상속 
class PartTimer(Student, Worker):
    def find_job(self):
        print('Find a job')

parttimer1 = PartTimer()    
parttimer1.sleep()   # sleep
parttimer1.study()   # Study hard
parttimer1.work()    # Work hard
parttimer1.find_job()    # Find a job
