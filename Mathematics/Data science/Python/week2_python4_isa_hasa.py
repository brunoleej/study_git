# is a has a
# 클래스를 설계하는 개념
# A is a B => A는 B이다. 상속을 이용해서 클래스를 만드는 방법
# A has a B => A는 B를 가진다. A가 B객체를 가지고 클래스를 만드는 방법
# 사람 클래스 생성 : 이름, 이메일, 정보출력

# is a
class Person:
    def __init__(self,name,email):
        self.name = name
        self.email = email

class Person2(Person):
    def info(self):
        print(self.name,self.email)

p = Person2('andy','andy@gmail.com')
p.info()    # andy andy@gmail.com

# has a
class Name:
    def __init__(self,name):
        self.name_str = name
class Email:
    def __init__(self,email):
        self.email_str = email

class hPerson:
    def __init__(self,name_obj,email_obj):
        self.name = name_obj
        self.email = email_obj
    def info(self):
        print(name.name_str,email.email_str)

name = Name('andy')
email = Email('andy@gmail.com')
p2 = hPerson(name,email)

p2.info()