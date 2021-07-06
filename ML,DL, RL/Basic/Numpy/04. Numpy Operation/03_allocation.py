import numpy as np

a = np.ones((2,3), dtype=int)
b = np.random.random((2,3))

a *= 3
print(a)
'''
[[3 3 3]
 [3 3 3]]
'''

b += a
print(b)
'''
[[3.74206601 3.3967647  3.01843652]
 [3.19417986 3.88757654 3.2331842 ]]
'''

a += b
'''
Traceback (most recent call last):
  File "c:\Vscode_Study\Study\Numpy\04. Numpy Operation\03_allocation.py", line 16, in <module>
    a += b
numpy.core._exceptions.UFuncTypeError: Cannot cast ufunc 'add' output from dtype('float64') to dtype('int32') with casting rule 'same_kind'
'''