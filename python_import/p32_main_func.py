import p31_sample
# import 되는 것 : 함수, 클래스, 변수
x = 222

def main_func() :
    print("x : ", x)

main_func()
# x :  222

p31_sample.test()
# import 되어 있는 변수를 그대로 가져온다. >> x :  111
