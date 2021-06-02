# lambda 함수 사용하기

gradient = lambda x : 2*x - 4

# 위 lamda와 동일한 뜻
def gradient2(x) :
    temp = 2*x - 4
    return temp

x = 3

# 결과 동일함
print(gradient(x))
print(gradient2(x))

# 2
# 2