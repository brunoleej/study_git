import numpy as np
# dtype을 따로 설정하지 않아도 자동으로 float 형태로 바꾸면서 배열을 만든다
e = [1,2,3,4,5]
print(np.asfarray(e))   # [1. 2. 3. 4. 5.]
print(np.asfarray(e).dtype) # float64
