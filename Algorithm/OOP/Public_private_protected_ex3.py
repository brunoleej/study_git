# 2. 학생 성적 관리 class 작성하기
    # attribute : 국어, 영어, 수학, 학생 이름 네 개의 속성
    # 생성자에서 각 속성을 객체 생성 시 전달된 인자값으로 설정
    # 각 속성은 private으로 설정
    # method : 전체 과목 점수 평균, 전체 과목 총점 두가지 method 구현
    # 각 method는 private으로 설정

class Student:
    def __init__(self, kor : int, eng : int, mat : int, name : str):
        self.__kor = kor
        self.__eng = eng
        self.__mat = mat
        self.__name = name
    
    def avg_student(self):
        return (self.__kor + self.__eng + self.__mat) / 3

    def sum_student(self):
        return self.__kor + self.__eng + self.__mat

student = Student(70, 80, 95, 'ji su')
print(student.avg_student())    # 81.66666666666667
print(student.sum_student())    # 245