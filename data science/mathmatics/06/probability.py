A = frozenset([])
B = frozenset(['H'])
C = frozenset(['T'])
D = frozenset(['H', 'T'])
P = {A: 0, B: 0.4, C: 0.6, D: 1}

print(P)
# {frozenset(): 0, frozenset({'H'}): 0.4, frozenset({'T'}): 0.6, frozenset({'T', 'H'}): 1}

