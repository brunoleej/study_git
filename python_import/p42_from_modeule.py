# machine 안에 있는 함수를 불러온다. 2가지 방법

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