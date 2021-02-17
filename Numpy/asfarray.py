import numpy as np
# dtype을 따로 설정하지 않아도
e = [1,2,3,4,5]
print(np.asfarray(e))   # [1. 2. 3. 4. 5.]
print(np.asfarray(e).dtype) # float64
