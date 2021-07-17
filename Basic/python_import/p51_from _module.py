# machine 안에 있는 함수를 불러온다. 3가지 방법

from machine.car import drive
from machine.tv import watch

drive()
watch()

print("===================")

# from machine import car
# from machine import tv
from machine import car, tv

car.drive()
tv.watch()

print("==========test==============")
# 1
from machine.test.car import drive
from machine.test.tv import watch

drive()
watch()

# 2 
from machine.test import car
from machine.test import tv

car.drive()
tv.watch()

# 3
from machine import test
test.car.drive()
test.tv.watch()

# 패키지의 단점 : 현재 파일이 있는 폴더와 같은 곳에만 들어가 실행할 수 있음
# 클래스 : 아나콘다, 어느 파일이든 적용할 수 있음