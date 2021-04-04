# 보간법 interpolate : 결측치를 채워준다.
# 결측치 수정하기 : 0으로 채운다. 중간값을 넣는다. 평균값을 넣는다. 위/아래에 있는 값을 그대로 넣는다. 행 삭제. + 보간법
# 보간 : 시계열, 연속된 데이터, 선형으로 되어 있는 데이터에서 사용하기 유용하다.

from pandas import DataFrame, Series
from datetime import datetime
import numpy as np
import pandas as pd

datestrs = ['3/1/2021', '3/2/2021','3/3/2021','3/4/2021','3/5/2021']
dates = pd.to_datetime(datestrs)
print(dates)
# 2021-03-01     1.0
# 2021-03-02     NaN
# 2021-03-03     NaN
# 2021-03-04     8.0
# 2021-03-05    10.0
print("======================================================")

ts = Series([1, np.nan, np.nan, 8, 10], index=dates)    # 위에 나온 날짜와 매칭된다.
print(ts)

# interpolate (보간)
ts_intp_linear = ts.interpolate()
print(ts_intp_linear)
# 2021-03-01     1.000000
# 2021-03-02     3.333333   >> 결측치를 채워준다.
# 2021-03-03     5.666667   >> 결측치를 채워준다.
# 2021-03-04     8.000000
# 2021-03-05    10.000000
