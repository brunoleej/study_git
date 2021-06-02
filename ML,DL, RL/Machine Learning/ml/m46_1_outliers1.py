# 이상치 처리
# 1. 0으로 처리
# 2. Nan으로 처리한 후, 보간
# 3. 등등

# Outlier
# "위치"로 잡는다.

import numpy as np
aaa = np.array([1,2,3,4, 6,7,90,100,5000,10000])


def outliers (data_out) :
    quartile_1, q2, quartile_3 = np.percentile(data_out, [25, 50, 75])  # 데이터의 25%, 50%, 75% 지점을 알려준다.  
    # np.percentile : 0을 최소값, 100을 최대값으로 백분율로 나타낸 특정 위치 값이다.

    print("1사분위 : ",quartile_1)  # 데이터의 25% 지점
    print("q2 :      ",q2)          # 중위값
    print("3사분위 : ",quartile_3)  # 데이터의 75% 지점
    iqr = quartile_3 - quartile_1   # 3분위 - 1분위 : 여기에 있는 데이터들을 정상데이터라고 봄
    lower_bound = quartile_1 - (iqr * 1.5)  # 정상데이터보다 1.5% 작은 데이터까지 정상데이터로 확장
    upper_bound = quartile_3 + (iqr * 1.5)  # 정상데이터보다 1.5% 큰 데이터까지 정상데이터로 확장
    return np.where((data_out > upper_bound) | (data_out < lower_bound))    # 최소의 값보다 작거나 최대의 값보다 큰 값을 이상치로 잡는다.


outlier_loc = outliers(aaa)
print("이상치 위치 : ", outlier_loc )

# 1사분위 :  3.25
# q2 :       6.5
# 3사분위 :  97.5
# 이상치 위치 :  (array([8, 9], dtype=int64),)

# 위 aaa 데이터를 boxplot으로 그린다. 
import matplotlib.pyplot as plt
plt.boxplot(aaa)
plt.show()