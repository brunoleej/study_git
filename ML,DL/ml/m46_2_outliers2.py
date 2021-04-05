# outliers1 >> 을 행렬형태도 적용할 수 있도록 수정

import numpy as np
import pandas as pd

# aaa = np.array([[1,2,3,4,10000,6,7,5000,90,100],
#                 [100,200,3,400,500,600,700,8,900,1000]])

aaa = np.array([[1,2,3,4,10000,6,7,5000,90,100],
                [1100,1200,3,1400,1500,1600,1700,8,1900,11000]])

aaa = aaa.transpose()
print(aaa.shape)    # (10, 2)

def outliers (data_out, column) :
    df = pd.DataFrame(data_out)

    quartile_1 = np.percentile(df[column].values, 25)
    quartile_3 = np.percentile(df[column].values, 75)
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    outlier_idx = df[column][ (df[column] < lower_bound) | (df[column] > upper_bound) ].index
    return outlier_idx

def outliers2 (data_out, column) :
    result = []
    for i in range(data_out.shape[1]) :
        q1, q2, q3 = np.percentile(data_out[:,i], [25, 50, 75])
        iqr = q3 - q1
        lower_bound = q1 - (iqr * 1.5)
        upper_bound = q1 + (iqr * 1.5)
        tmp = np.where((data_out[:,i] > upper_bound) | (data_out[:,i] < lower_bound))
        result.append(tmp)
    return np.array(result)


outlier_idx = outliers(aaa,0)
print(outlier_idx)
outlier_idx2 = outliers(aaa,1)
print(outlier_idx2)

# Int64Index([4, 7], dtype='int64')
# Int64Index([2, 7, 9], dtype='int64')


outlier_idx = outliers2(aaa,0)
print(outlier_idx)

# 위 aaa 데이터를 boxplot으로 그린다. 
import matplotlib.pyplot as plt
plt.boxplot(aaa)
plt.show()