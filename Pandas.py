# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 17:46:18 2020

@author: lucifelex
"""


"""
# 讀取檔案
# read_excel
# read_csv
# read_html

# 寫入檔案
# ExcelWriter
import pandas as pd
#data = pd.read_csv('ExpensesRecord.csv')
df = pd.read_excel('ExpensesRecord.xls', 'sheet')
#data = pd.read_html('http://www.fdic.gov/bank/individual/failed/banklist.html')
print(df.head(5) )      #顯示前五筆資料

from pandas import ExcelWriter
writer = ExcelWriter('test.xlsx', engine='xlsxwriter')      #1st 儲存檔名, 2nd 引擎
df.to_excel(writer, sheet_name='sheet2')        #1st 目的檔案, 2nd sheet名稱
writer.save()       #儲存檔案
"""



"""
# 讀取網路上的表格
import pandas as pd

df = pd.read_html('http://www.fdic.gov/bank/individual/failed/banklist.html')
print(df[0].head(5) )
"""


"""
# DataFrame 資料格式
import pandas as pd
import numpy as np
DataFrame = pd.read_csv('ExpensesRecord.csv')
print(DataFrame["說明"])          #欄位
print(DataFrame[["說明","支出金額"]] )        #兩個欄位

# 字典型態的 DataFrame
df = pd.DataFrame({'Math': [90, 91,92, 93, 94],'English': np.arange(80,85,1) })
print(df[["Math","English"]])
"""



"""
# DataFrame計算
import pandas as pd
DataFrame = pd.read_csv('ExpensesRecord.csv')
# 用計算結果新增一個欄位
DataFrame["單價"]=DataFrame["支出金額"]/DataFrame["數量"]
print(DataFrame[["數量","支出金額","單價"]] )
"""



"""
# Yahoo 股票 API
# get_data_yahoo
import pandas as pd
pd.core.common.is_list_like = pd.api.types.is_list_like

from pandas_datareader import data, wb
import pandas_datareader.data as web


import yfinance as yf    # Yahoo 股票 API
yf.pdr_override()

df = web.get_data_yahoo("AAPL", start="2018-01-01", end="2018-12-02")
print(df.head())
writer=pd.ExcelWriter('AAPL.xlsx')
df.to_excel(writer,'AAPL')
writer.save()


from pandas import ExcelWriter
writer = ExcelWriter('testaapl.xlsx', engine='xlsxwriter')
df.to_excel(writer, sheet_name='sheet2')

df.to_csv("testaapl.csv")
"""



"""
# 篩選 DataFrame資料
# DatetimeIndex 將時間轉換成 DataFrame的時間格式
import pandas as pd
df = pd.read_excel('AAPL.xlsx', 'AAPL')
print(df.head())        #[5 rows x 7 columns]
print(type(df))     #<class 'pandas.core.frame.DataFrame'>

# 2 data info
print(df.shape)     #(232, 7)
print(df.columns)#Index(['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'], dtype='object')
print(df.index)     #RangeIndex(start=0, stop=232, step=1)
print(df.info())
print(df.describe())


# 3 filter'

print("--------------------")
print(df['Date'] == '2018-01-05')
print(df[df['Date'] == '2018-01-05'])       #顯示特定日期的資料
print(df[(df['Date'] >= '2018-07-05') & (df['Date'] <= '2018-07-10' )])     #顯示特定區間時間的資料
print(df[df['Open'] > 194.2])       #找出開盤價高於 194.2的資料
print(df[['Date','Open']])
print(df[['Date','Open']][:5])      #顯示前5筆資料
print(df.sort_values(by=['Volume'])[:5])    #增序排列 顯示前五筆
print(df.sort_values(by=['Volume'], ascending=False)[:5])   #減序排列 顯示前五筆
print(df['Open'][:30].rolling(7).mean())    #處理前30筆資料 每7筆資料的平均開盤價


# 4 Calculation
print("--------------------")
df['diff'] = df['Close']-df['Open']    #股價每天的浮動
df['year'] = pd.DatetimeIndex(df['Date']).year    #將時間轉換成 DataFrame的時間格式 並取得年分
df['month'] = pd.DatetimeIndex(df['Date']).month  #將時間轉換成 DataFrame的時間格式 並取得月分
print(df.head())
print("April Volume sum=%.2f" % df[df['month'] == 4][['Volume']].sum())    #取得4月份的全部交易量總和
print("April Open mean=%.2d" % df[df['month'] == 4][['Open']].mean())    #取得4月份的平均開盤價
"""

#  報錯
#  5 matplotlib
import matplotlib.pyplot as plt
df.plot(x='Date', y='Open',grid=True, color='blue')
plt.show()


import matplotlib.pyplot as plt
df.plot( y='diff',grid=True, color='red',kind='hist')
plt.show()

fig, ax = plt.subplots()
for name, group in df.groupby('month'):
    group.plot(x='day', y='Open', ax=ax, label=name)
plt.show()

fileds=['Open','Close','High']
fig, ax = plt.subplots()
for name in fileds:
    df.plot(x='Date', y=name, ax=ax, label=name)
plt.show()

dfMonths = df.loc[df['month'].isin([1,2,3,4,5,6,7])]
print(dfMonths)
dfMonthsPivot = dfMonths.pivot_table(values = 'High', columns = 'month', index = 'day')
dfMonthsPivot.plot(kind = 'box',title = 'Months High')
plt.show()









