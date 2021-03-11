A = frozenset([])
B = frozenset(['H'])
C = frozenset(['T'])
D = frozenset(['H', 'T'])

print(set([A, B, C, D]))
# {frozenset(), frozenset({'H'}), frozenset({'T'}), frozenset({'T', 'H'})}