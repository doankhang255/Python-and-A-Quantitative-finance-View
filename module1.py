import pandas as pd
import pandas
import matplotlib.pyplot as plt

ms = pd.read_csv(r'C:\Users\doank\OneDrive\Personal Vault\Python and Statistics for Financial Analysis\microsoft.csv', index_col = 0)
ms.index = pd.to_datetime(ms.index)
# run this cell to ensure Microsoft's stock data is imported
# print(ms.index[-1])
# print(type(ms.index[-1]))
# print(ms.head(6))
print(ms.shape) #number of row and column
# print(ms.tail(5))
print(ms.describe())
# print(ms.loc[:,"Open"])
# print(ms.loc['2015-01-1':'2015-01-28', 'Open'])
# print(ms.iloc[1,2])  #print(repr(ms.iloc[1, 4]))     # hiển thị đầy đủ chính xác
# print(ms.iloc[777: , :])
# ms.loc['2015-01-02':'2016-01-02', 'Open'].plot()
# ms.loc['2016-01-02':'2017-01-02', 'Open'].plot()
# ms.loc['2017-01-02':'2017-12-31', 'Open'].plot()
# plt.xlabel("Datetime")
# plt.ylabel("Open Price")
# plt.legend(['2015', '2016', '2017']) # bảng chú thích
# plt.show()
# print(ms[['Open','High','Close']])
# ... matrix = [[i*j for j in range(3)] for i in range(3)] ... # nested comprehension 
# ... print(matrix)  # ➝ [[0, 0, 0], [0, 1, 2], [0, 2, 4]] ...

# ms["price1"] = ms['Close'].shift(-1)     
# ms["pricedif"] = ms["price1"] - ms['Close']
# ms["Daily Return"] = ms["pricedif"] / ms['Close']
# ms["direction"] = [ 1 if ms.loc[ ei ,"pricedif"] > 0 else -1 for ei in ms.index ]
# ms["MA"] = ms["Close"].rolling(4) . mean()
# ms["Average"] = (ms["Close"] + ms["Close"].shift(1) + ms["Close"].shift(2))/3
# print(ms.iloc[0 : 4 , 6])
# print (ms["direction"]) 
# print (ms["MA"])
ms["MA10"] = ms["Close"].rolling(10).mean()
ms["MA50"] = ms["Close"].rolling(50).mean()
ms["Shares"] = [1 if ms.loc[ei , "MA10"] > ms.loc[ei, "MA50"] else 0 for ei in ms.index]
ms["Close1"] = ms["Close"].shift(-1)
ms["Profit"] = [ms.loc[ei,"Close1"] - ms.loc[ei, "Close"] 
                if ms.loc[ei, "Shares"] == 1
                else 0 
                for ei in ms.index ]
ms["Profit"].plot()
plt.axhline(y=0, color = "red")
# plt.show()

# ms["wealth"] = ms["Profit"].cumsum() #cumulative wealth
# ms.tail()
# ms["wealth"].plot(color = "blue")
# plt.show()
print(ms[["Open","Low"]])