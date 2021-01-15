# Series index와 value로 이루어진 데이터 타입
# 동일한 데이터 타입의 값을 가짐
# value만 설정하면 index는 0부터 자동으로 설정 됨
import numpy as np
import pandas as pd
np.random.seed(1234)

series = pd.Series(np.random.randint(10,size=5))
print(series)
'''
0    5
1    9
2    9
3    0
4    6
'''

# index 설정
series = pd.Series(np.random.randint(10,size = 5), index = list('ABCDE'))
print(series)
'''
A    4
B    8
C    8
D    2
E    0
'''
print(series.index, series.values)  # Index(['A', 'B', 'C', 'D', 'E'], dtype='object') [3 4 8 3 6]

# index로 value값 확인
print(series['B'],series.B)   #  인덱스가 아닌 경우 value.B 이런식으로 확인이 가능, 9 9

# Series value 변경
series['C'] = 10
print(series)
'''
A     1
B     4
C    10
D     0
E     5
'''

# Broadcasting
print(series * 10)
'''
A     90
B     10
C    100
D     90
E     60
'''

# 다중 인덱스 출력
print(series[['B','E']])
'''
B    1
E    6
'''

# offset index
print(series[2::2]) # 2에서 끝까지인데 2칸씩 건너뛰면서
'''
C    10
E     6
'''
print(series[::-1]) # 역순으로 출력
'''
E     6
D     9
C    10
B     1
A     9
'''

# Series Calculate
series2 = pd.Series({'B':3,"E":5,'F':7})
print(series2)
'''
B    3
E    5
F    7
'''

# add
result = series + series2   # 같은 인덱스끼리 연산이 됨
print(result)   # None
'''
A     NaN
B     4.0
C     NaN
D     NaN
E    11.0
F     NaN
'''

print(result.isnull())
'''
A     True
B    False
C     True
D     True
E    False
F     True
'''
print(result[result.isnull()])
'''
A   NaN
C   NaN
D   NaN
F   NaN
'''
result[result.isnull()] = series
print(result)   # F값은 series에 원래 존재하지 않으므로 NaN값 들어감
'''
A     9.0
B     4.0
C    10.0
D     9.0
E    11.0
F     NaN
'''
print(result.isnull)
''''
B     4.0
C    10.0
D     9.0
E    11.0
F     NaN
'''
result[result.isnull()] = series2
print(result)   # series2 값이 들어감
'''
A     9.0
B     4.0
C    10.0
D     9.0
E    11.0
F     7.0
'''