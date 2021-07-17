# 다중 상속된 클래스의 나열 순서가 자식 클래스가 속성(멤버변수, 메소드) 호출에 영향을 줌
    # 상속된 클래스 중 앞에 나열된 클래스부터 속성을 찾음

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
class PartTimer(Student, Worker):
    def find_job(self):
        print('Find a job')

# 아래의 코드를 실행시키면서 Multiple Inheritance.py 파일의 결과물과 비교하며, 출력 값이 왜 이렇게 나왔는지 이해하기(다중 상속)
parttimer1 = PartTimer()
parttimer1.study()  # Study hard
parttimer1.work()   # Work hard
parttimer1.play()   # play with friends --> 앞에 나열된 play를 실행