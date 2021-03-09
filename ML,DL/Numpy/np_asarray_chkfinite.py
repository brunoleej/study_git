import numpy as np
# 배열로 만들려고 하는 데이터 input에 결측값(NaN)이나 무한수(infinite number)가 들어잇을 경우 'ValueError'를 반환하게 한다.

# 결측치가 무한수가 존재할 경우 오류를 반환한다.
d = [1,2,3,4, np.nan]

print(np.asarray_chkfinite(d, dtype = float))
# ValueError: array must not contain infs or NaNs

