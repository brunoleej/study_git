# Saladent(공부하는 직장인) 클래스 만들기 해당 객체는 Worker와 Student 클래스를 
# 상속 받고, play() 메소드 호출 시 'drinks alone'이 출력되도록 할 것

# 클래스 선언
class Person:
    def sleep(self):
        print('sleep')

class Student(Person):
    def study(self):
        print('Study hard')
    
    def play(self):
        print('play with friends')

class Worker(Person):
    def work(self):
        print('Work hard')

    def play(self):
        print('drinks alone')

# 다중 상속 
class Saladent(Worker,Student):
    pass

saladent1 = Saladent()
saladent1.play()    # drinks alone