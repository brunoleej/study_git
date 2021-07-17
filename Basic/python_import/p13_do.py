import p11_car
# 실행 >> P11_car.py의 module 이름은 :  p11_car >> module 이름이 main이 아니라 불러온 파일의 파일이름으로 바뀜

import p12_tv
# 실행 >> P12_tv.py의 module 이름은 :  p12_tv

print("========================")
print("p13_do.py의 module 이름은 :", __name__)
# p13_do.py의 module 이름은 : __main__
print("========================")

p11_car.drive()
p12_tv.watch()
