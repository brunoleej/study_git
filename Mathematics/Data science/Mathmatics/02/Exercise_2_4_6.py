# 보스턴 집값 문제를 선형 예측모형 Xw = y_hat로 풀었을 때의 가중치벡터 를 최소 자승 방법으로 구하라. 
# 행렬과 벡터 데이터는 다음과 같이 얻을 수 있다.
import numpy as np
from sklearn.datasets import load_boston

boston = load_boston()
X = boston.data
y = boston.target

'''
1. CRIM: 범죄율
2. INDUS: 비소매상업지역 면적 비율
3. NOX: 일산화질소 농도
4. RM: 주택당 방 수
5. LSTAT: 인구 중 하위 계층 비율
6. B: 인구 중 흑인 비율
7. PTRATIO: 학생/교사 비율
8. ZN: 25,000 평방피트를 초과 거주지역 비율
9. CHAS: 찰스강의 경계에 위치한 경우는 1, 아니면 0
10. AGE: 1940년 이전에 건축된 주택의 비율
11. RAD: 방사형 고속도로까지의 거리
12. DIS: 보스톤 직업 센터 5곳까지의 가중평균거리
13. TAX: 재산세율
'''
x, resid, rank, s = np.linalg.lstsq(X,y)
print(x)