# 현재 파일의 하단 폴더에 있는 모듈을 불러오고 싶음
# 패키지 : import 할 수 있는 덩어리들을 가리킴
# 모듈 : 패키지 가장 하단에 있는 것들
import machine.car  # machine폴더 의 car.py파일을 불러옴
import machine.tv

machine.car.drive()
machine.tv.watch()