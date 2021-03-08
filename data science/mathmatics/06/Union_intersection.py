A1 = set([1, 2, 3, 4])
A2 = set([2, 4, 6])
A3 = set([1, 2, 3])
A4 = set([2, 3, 4, 5, 6])

print(A1.union(A2))
print(A2 | A1)
print(A3.intersection(A4))
print(A4 & A3)