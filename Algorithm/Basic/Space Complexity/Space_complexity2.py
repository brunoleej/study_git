# 공간 복잡도 예제2
# n! 팩토리얼 구하기
    # n! = 1 x 2 x ... x n
# 재귀함수를 사용하였으므로, n에 따라, 변수 n이 n개가 만들어지게 됨
    # factorial 함수를 재귀 함수로 1까지 호출하였을 경우, n부터 1까지 스택이 쌓이게 됨
# 공간 복잡도는 O(n)

def factorial(n):
    if n > 1:
        return n * factorial(n - 1)
    else:
        return 1

print(factorial(3)) # 6