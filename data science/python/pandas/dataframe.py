# DataFrame은 여러개의 Series로 구성이 되어 있음
# 같은 Column에 있는 value값은 같은 데이터 타입을 갖습니다.
import numpy as np
import pandas as pd

# DataFrame 생성 1(Dictionary의 list)
datas = {
    "name":['dss','camp'],
    'email':['dss@gmail.com','camp@daum.net']
}
print(datas)    # {'name': ['dss', 'camp'], 'email': ['dss', 'camp@daum.net']}

df = pd.DataFrame(datas)
print(df)
'''
   name          email
0   dss  dss@gmail.com
1  camp  camp@daum.net
'''

# DataFrame 생성 2(list의 Dictionary)
datas = [
    {"name":'dss','email':'dss@gmail.com'},
    {"name":'camp','email':'camp@daum.com'}
]
print(datas)    # [{'name': 'dss', 'email': 'dss@gmail.com'}, {'name': '', 'email': 'dss@gmail.com'}]

df = pd.DataFrame(datas)
print(df)

# index 추가
df = pd.DataFrame(datas,index=['one','two'])
print(df)
'''
     name           email
one   dss   dss@gmail.com
two  camp  camp@gmail.com
'''
print(df.index) # Index(['one', 'two'], dtype='object')
print(df.columns)   # Index(['name', 'email'], dtype='object')
print(df.values)
'''
[['dss' 'dss@gmail.com']
 ['camp' 'camp@gmail.com']]
 '''

# DataFrame에서 Data의 선택 => row, column,(row,column)
df = pd.DataFrame(datas)
print(df)
'''
   name           email
0   dss   dss@gmail.com
1  camp  camp@gmail.com
'''
print("===============")
print(df.loc[1])   # row 선택
'''
name               camp
email    camp@gmail.com
'''
print("===============")
print(df.loc[1]['email'])   # camp@gmail.com

# index가 있으면 수정, 없으면 추가
df.loc[2] = { "name":'andy','email':'andy@naver.com'}
print(df)
'''
   name           email
0   dss   dss@gmail.com
1  camp  camp@gmail.com
2  andy  andy@naver.com
'''

df['id'] = ''
print(df)
'''
   name           email id
0   dss   dss@gmail.com   
1  camp  camp@gmail.com   
2  andy  andy@naver.com 
'''

df['id'] = range(1,4) # np.arange(1,4)
print(df)
'''
   name           email  id
0   dss   dss@gmail.com   1
1  camp  camp@gmail.com   2
2  andy  andy@naver.com   3
'''

# row, column 선택
print('==============')
print(df.loc[[0,2],['email','id']])    #  value.loc[[row,column]]
'''
            email  id
0   dss@gmail.com   1
2  andy@naver.com   3
'''

# Column Data 순서 설정
print(df)
'''
   name           email  id
0   dss   dss@gmail.com   1
1  camp  camp@gmail.com   2
2  andy  andy@naver.com   3
'''

print('==================')
df = df[['id','name','email']]
print(df)
'''
   id  name           email
0   1   dss   dss@gmail.com
1   2  camp  camp@gmail.com
2   3  andy  andy@naver.com
(base) jisu@jisuui-MacBookPr
'''

# head,tail
print(df.head())    # Default : 5
# print(df.head(n=3)) # 설정 가능

print(df.tail())    # Default : 5
# print(df.tail(n=3))

print('===========================')
print(df)
'''
   id  name           email
0   1   dss   dss@gmail.com
1   2  camp  camp@gmail.com
2   3  andy  andy@naver.com
'''

df['domain'] = df['email'].apply(lambda email : email.split('@')[1].split('.')[0])
print('===========================')
print(df)
'''
   id  name           email domain
0   1   dss   dss@gmail.com  gmail
1   2  camp   camp@daum.com   daum
2   3  andy  andy@naver.com  naver
'''
