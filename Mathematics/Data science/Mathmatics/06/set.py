A = set([1, 2, 3, 3, 2]) # 중복된 자료는 없어진다.
B = frozenset(['H', 'T'])
C = {"\u2660", "\u2661", "\u2662", "\u2663"}

print(A)
print(B)
print(C)
print(type(C))