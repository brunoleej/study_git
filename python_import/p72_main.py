import p71_byunsu as p71

print(p71.aaa)          # aaa = 2 임포트
print(p71.square(10))   # 제곱 1024

print("===============")

from p71_byunsu import aaa, square   
aaa = 3
print(aaa)           # aaa = 3 
print(square(10))    # 임포트한 3을 댕겨옴 >> 제곱 1024    