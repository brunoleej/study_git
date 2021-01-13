import numpy as np
import pandas as pd

prices = pd.read_csv('./삼성전자.csv', index_col=0, header=0, encoding='cp949')  

# Preprocessing
prices['시가'] = prices['시가'].str.replace(pat=r'[^\w]', repl=r'', regex=True)
prices['고가'] = prices['고가'].str.replace(pat=r'[^\w]', repl=r'', regex=True)
prices['저가'] = prices['저가'].str.replace(pat=r'[^\w]', repl=r'', regex=True)
prices['종가'] = prices['종가'].str.replace(pat=r'[^\w]', repl=r'', regex=True)
prices['거래량'] = prices['거래량'].str.replace(pat=r'[^\w]', repl=r'', regex=True)
prices['금액(백만)'] = prices['금액(백만)'].str.replace(pat=r'[^\w]', repl=r'', regex=True)
prices['개인'] = prices['개인'].str.replace(pat=r'[^\w]', repl=r'', regex=True)
prices['기관'] = prices['기관'].str.replace(pat=r'[^\w]', repl=r'', regex=True)
prices['외인(수량)'] = prices['외인(수량)'].str.replace(pat=r'[^\w]', repl=r'', regex=True)
prices['외국계'] = prices['외국계'].str.replace(pat=r'[^\w]', repl=r'', regex=True)
prices['프로그램'] = prices['프로그램'].str.replace(pat=r'[^\w]', repl=r'', regex=True)

# to_numeric
for idx in prices.columns:
    prices[idx] = pd.to_numeric(prices[idx])
    
prices_sort = prices.sort_values(by='일자' ,ascending=True) 
print(prices_sort.head())

y = prices_sort.iloc[:,3:4]
print(y)

del prices_sort['종가']
prices_sort['종가'] = y 
print(prices_sort)
print(prices_sort.columns)

print(prices_sort.isnull().sum())    
# null : 2018-04-30, 2018-05-02, 2018-05-03 >> 거래량  3 / 금액(백만) 3
df_dop_null = prices_sort.dropna(axis=0)
print(df_dop_null.shape)    # (2397, 14)

# 액면가 조정 (시가, 고가, 저가, 종가, 거래량, 금액, 개인, 기관, 외인, 외국계, 프로그램)
# 시가
a = df_dop_null.iloc[:1735,:1] / 50
b = df_dop_null.iloc[1735:,:1]
df_dop_null['시가'] = pd.concat([a,b])

# 고가
a = df_dop_null.iloc[:1735,1:2] / 50
b = df_dop_null.iloc[1735:,1:2]
df_dop_null['고가'] = pd.concat([a,b])

# 저가
a = df_dop_null.iloc[:1735,2:3] / 50
b = df_dop_null.iloc[1735:,2:3]
df_dop_null['저가'] = pd.concat([a,b])

# 거래량
a = df_dop_null.iloc[:1735,4:5] * 50
b = df_dop_null.iloc[1735:,4:5]
df_dop_null['거래량'] = pd.concat([a,b])

# 종가
a = df_dop_null.iloc[:1735,13:14] / 50
b = df_dop_null.iloc[1735:,13:14]
df_dop_null['종가'] = pd.concat([a,b])

print(df_dop_null)

# 5. 상관계수 확인
print(df_dop_null.corr())
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1.0) # 폰트 크기
sns.heatmap(data=df_dop_null.corr(),square=True, annot=True, cbar=True)
plt.show()
# 상관계수 0.5 이상 : 시가, 고가, 저가, 종가, 금액, 기관, 외인비

del df_dop_null['등락률']
del df_dop_null['거래량']
del df_dop_null['금액(백만)']
del df_dop_null['개인']
del df_dop_null['기관']
del df_dop_null['외인(수량)']
del df_dop_null['외국계']
del df_dop_null['프로그램']

print(df_dop_null)   # [2397 rows x 6 columns]

# 최종 데이터 확인 
print(df_dop_null.shape) # (2397, 6)

# numpy 저장 
final_prices = df_dop_null.to_numpy()
print(final_prices)
print(type(final_prices)) # <class 'numpy.ndarray'>
print(final_prices.shape) # (2397, 6)
np.save('./samsung_prices.npy', arr=final_prices)