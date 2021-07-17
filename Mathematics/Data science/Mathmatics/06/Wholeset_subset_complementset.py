A1 = set([1, 2, 3, 4])
A2 = set([2, 4, 6])
A3 = set([1, 2, 3])
A4 = set([2, 3, 4, 5, 6])

print(A3.issubset(A1))
print(A3 <= A1)
print(A3.issubset(A2))
print(A3 <= A2)
print(A3 <= A3)    # 모든 집합은 자기 자신의 부분집합이다
print(A3 < A3)  # 모든 집합은 자기 자신의 진부분집합이 아니다.