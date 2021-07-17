import matplotlib
from matplotlib import font_manager, rc
import platform

 

try:
    if platform.system() == 'Windows':
        #윈도우인 경우
        font_name = font_manager.FontProperties(fname="c:/Windows/fonts/malgun.ttf").get_name()
        rc('font', family=font_name)
    else:
        #Mac인 경우
        rc('font', family='AppleGothic')
except:
    pass
matplotlib.rcParams['axes.unicode_minus'] = False
