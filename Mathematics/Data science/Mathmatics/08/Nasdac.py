import pandas_datareader.data as web
symbol = "NASDAQCOM"
data = pd.DataFrame()
data[symbol] = web.DataReader(symbol, data_source="fred", start="2009-01-01", end="2018-12-31")[symbol]
data = data.dropna()
data.plot(legend=False)
plt.xlabel("날짜")
plt.title("나스닥 지수")
plt.show()