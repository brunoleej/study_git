# 환경변수 상에 path가 걸려있는 것들 중 : anaconda3 에 임의로 만들어 두었던 폴더, 파일을 불러옴

from test_import import p62_import

# 임포트한 파일 안에 있는 함수를 불러올 수 있음
p62_import.sum2()

print("===================")

from test_import.p62_import import sum2
sum2()
