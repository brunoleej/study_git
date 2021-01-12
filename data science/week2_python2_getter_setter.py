# class의 getter, setter
# 객체의 내부 변수에 접근할 때 특정 로직을 거쳐서 접근시키는 방법
class User:
    def __init__(self,first_name):
        self.first_name = first_name
    
    def setter(self,first_name):
        if len(first_name) >= 3:
            self.first_name = first_name
        else:
            print('error')

    def getter(self):
        print('getter')
        return self.first_name.upper()
   
    name = property(getter,setter)

user1 = User("andy")

print(user1.first_name) # andy 

# getter 함수 실행
print(user1.name)   # getter andy

# setter 함수 실행
user1.name = "jhon"
print(user1.name)   # JHON