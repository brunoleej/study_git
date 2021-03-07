import cvxpy as cp

# 변수의 정의
a = cp.Variable() # A의 생산량
b = cp.Variable() # B의 생산량

# 조건의 정의
constraints = [
 a >= 100, # A를 100개 이상 생산해야 한다.
 b >= 100, # B를 100개 이상 생산해야 한다. 
 a + 2 * b <= 500, # 500시간 내에 생산해야 한다.
 4 * a + 5 * b <= 9800, # 부품이 9800개 밖에 없다.
]

# 문제의 정의
obj = cp.Maximize(3 * a + 5 * b)
prob = cp.Problem(obj, constraints)

# 계산
prob.solve()

# 결과
print("상태:", prob.status)
print("최적값:", a.value, b.value)
