import pandas as pd

df = pd.DataFrame([1,2,3,4],[4,5,6,7],[7,8,9,10],columns = list('abcd'),index=list(('가나다라')))
print(df)