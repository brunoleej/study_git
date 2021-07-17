import numpy as np

sp.random.seed(0)
x = sp.stats.norm(0, 2).rvs(1000) # 평균=0, 표준편차=2 인 정규분포 데이터 생성

print(np.var(x), np.std(x)) # 편향 표본분산, 표본표준편차
print(np.var(x, ddof=1), np.std(x, ddof=1)) # 비편향 표본분산, 표본표준편차