# First-class 함수 : 함수 자체를 인자로 다른 함수에 전달. 다른 함수의 결과값으로 리턴. 함수를 변수에 할당할 수 있는 함수
    # 사실 파이썬에서는 모든 것이 객체!
    # 파이썬 함수도 객체로 되어 있어서, 기본 함수 기능 이외 객체와 같은 활용이 가능 (파이썬 함수들은 First-class 함수로 사용 가능)
    # 지금까지 배운 언어의 맥락과는 뿌리가 다른 사고 - 함수형 프로그래밍에서부터 고안된 기법
# 다른 변수에 함수 할당 가능

def calc_square(digit):
    return digit * digit

print(calc_square(2))   # 4

# 1. func1이라는 변수에 함수를 할당 가능
func1 = calc_square
print(calc_square)  # <function calc_square at 0x000001E3096E7160>

print(func1(2)) # 4