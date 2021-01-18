# 한글화 지원 모듈
import matplotlib.pyplot as plt
from matplotlib import rc

rc('font', family='AppleGothic')
plt.rcParams['axes.unicode_minus'] = False

# Example
# 패키지 임포트
from scipy import misc

# 컬러 이미지 로드
img_rgb = misc.face()

# 데이터의 모양
img_rgb.shape
rc('font', family='AppleGothic')
plt.rcParams['axes.unicode_minus'] = False

plt.figure(figsize = (15,10))

plt.subplot(221)
plt.imshow(img_rgb,cmap = plt.cm.gray) # 컬러 이미지 출력
plt.axis('off')
plt.title('RGB 컬러 이미지')

plt.subplot(222)
plt.imshow(img_rgb[:,:,0],cmap=plt.cm.gray) # red 채널 출력
plt.axis('off')
plt.title('Red 채널')

plt.subplot(223)
plt.imshow(img_rgb[:,:,1],cmap = plt.cm.gray) # green 채널 출력
plt.axis('off')
plt.title('Green 채널')

plt.subplot(224)
plt.imshow(img_rgb[:,:,2],cmap = plt.cm.gray) # blue 채널 출력
plt.axis('off')
plt.title('Blue 채널')
plt.show()