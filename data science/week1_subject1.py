# odd_even 함수를 만들어 주세요. 
# 숫자형 값을 두 개 받을 수 있게 해주세요.
# 받은 두 숫자의 제곱을 각각 새로운 변수로 만들어 저장하세요. 
# odd_even 함수내에 조건문을 주어 새롭게 만든 변수 2개의 합이 짝수일 경우 '짝수'라는 문자열을 , 홀수일 경우 '홀수'라는 문자열을 반환하도록 해주세요.

a, b = map(int,input("Insert Number: ").split(','))
result = []

def odd_even(a,b):
  result.append(a**2)
  result.append(b**2)
  if a + b %2 == 0:
    print("짝수")
  else:
    print('홀수')
  return result
  
print(result)
print(odd_even(2,5))