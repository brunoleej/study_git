# EllipticEnvelope
# outlier와의 차이점 : 
    ## EllipticEnvelope는 가우스 분포 & 공분산를 사용한다. 장점 2차 행렬도 사용할 수 있다.
    ## outlier : 퍼센트로 사용한다.

from sklearn.covariance import EllipticEnvelope
import numpy as np

# aaa = np.array([[1,2,-10000,3,4,6,7,8,90,100,5000]])
aaa = np.array([[1,2,10000,3,4,6,7,8,90,100,5000],
                [1100,1200,3,1400,1500,1600,1700,8,1900,11000,1001]])
aaa = np.transpose(aaa)
print(aaa.shape)    # (11, 2)

outlier = EllipticEnvelope(contamination=.2)
# contamination : 전체 데이터의 몇 퍼센트의 오염도, 몇 퍼센트를 아웃라이어로 잡을 것인가 (0.1이 디폴트)
outlier.fit(aaa)

print(outlier.predict(aaa))
# [ 1  1 -1  1  1  1  1  1  1  1  1] .1
# [ 1  1 -1  1  1  1  1  1  1 -1 -1] .2
# [ 1  1 -1  1  1  1  1  1  1 -1 -1] .3
# 두 열 중 하나라도 이상치가 있을 때 그 이상치의 위치를 표시해준다.
