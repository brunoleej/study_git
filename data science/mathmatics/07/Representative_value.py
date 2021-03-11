import numpy as np

x = np.random.normal(size=21)

print("표본평균 = {}, 표본중앙값 = {}".format(np.mean(x), np.median(x)))
# 표본평균 = 0.01299682800609689, 표본중앙값 = -0.25700440594140933

bins = np.linspace(-4, 4, 17)

ns, _ = np.histogram(x, bins=bins)
m_bin = np.argmax(ns)
print("최빈구간 = {}~{}".format(bins[m_bin], bins[m_bin + 1]))
# 최빈구간 = 0.0~0.5