# 2. func1 이라는 변수는 calc_square 함수를 가리키고, calc_square 와 마찬가지로 인자도 넣어서 결과도 얻을 수 있음 (완전 calc_square와 동일)

print(func1)    # 4
func1(2)    # <function calc_square at 0x000001C5A3627160>

class MyClass:
    def my_class(self):
        print('안녕')
        pass

object1 = MyClass()
my_class1 = object1.my_class
my_class1() # 안녕
