# import p31_sample
# import 되는 것 : 함수, 클래스, 변수
# from을 사용해서 그 모듈 안에 있는 함수를 불러올 수 있음

from p31_sample import test

x = 222

def main_func() :
    print("x : ", x)

main_func()
# x :  222

# p31_sample.test()
# import 되어 있는 변수를 그대로 가져옴. >> x :  111
# 대신,

# test를 import 했으므로 그대로 사용할 수 있음
test()