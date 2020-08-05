# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 07:55:05 2020

@author: lucifelex
"""

#網路功能

import urllib.request  
import sys
from xml.etree import ElementTree
import json
import ssl

"""
try :       #處理網路連線正常
    url = "http://www.powenko.com/download_relesase/get.php?name=powenko"
    req = urllib.request.Request(url)
    response = urllib.request.urlopen(req)    
    if response.code==200 :        
        contents = response.read().decode(response.headers.get_content_charset())
        print(contents)
    else:        
        print(respose.code)        
except:
    print('error')
"""        


"""
try:    # XML資料
    url="http://data.taipei/opendata/datalist/datasetMeta/download?id=5bc82dc7-f2a2-4351-abc8-c09c8a8d7529&rid=1f1aaba5-616a-4a33-867d-878142cac5c4"
    req = urllib.request.Request(url)
    response = urllib.request.urlopen(req)
    if response.code == 200:
        contents = response.read().decode(response.headers.get_content_charset())
        print(contents)
    else:
        print(response.code)
except:
    print('error')
"""    


"""
try:    #取得網站回應資料  POST
    url = "http://www.powenko.com/download_release/post.php"
    values = {'name':'poweko','password':123}
    data = urllib.parse.urlencode(values)
    data = data.encode('utf-8')
    req = urllib.request.Request(url,data)      #向網站submit input
    with urllib.request.urlopen(req) as response:
        contents = response.read().decode(response.headers.get_content_charset())
        print(contents)    
    
except:
    print('error')

"""


"""
# XML 找出特定資料
def print_node(node):
    try:
        print("node.text: %s" % node.text)      #逗號?
    except:
        print("node.text : null")

try:
    url = "http://data.taipei/opendata/datalist/datasetMeta/download?id=ece023db-a5f8-4399-97da-f04d7f4009e3&rid=1a2d417e-c121-4a12-835f-97ee6852c4b8"
    req = urllib.request.Request(url)
    response = urllib.request.urlopen(req)
    if response.code == 200:
        contents = response.read().decode(response.headers.get_content_charset())
        root = ElementTree.fromstring(contents)
        lst_node = root.findall("MAP/PERSON_IN_CHARGE")     #找所有標記 MAP內 標記為 PERSON_IN_CHARGE的項目
        for node in lst_node:
            print_node(node)
except:
    print('error')
"""


"""
# JSON
data = {"name": "Powen Ko", "shares": 100, "price": 542.23}

json_str = json.dumps(data)     #將字典資料轉換成JSON純文字資料
print (json_str)

data = json.loads(json_str)     #將JSON純文字資料轉換成字典
print(data)

with open('data.json','w') as f:        #開啟data.json 檔案,準備寫入
    json.dump(data, f)          #將data資料,寫入data.json檔案
    
with open('data.json','r') as f:        #讀取data.json 檔案
    data = json.load(f)
    print('=====')
    print(data)
"""    



"""
# 無法執行
context = ssl._create_unverified_context()      #設定SSL
url="http://data.taipei/opendata/datalist/datasetMeta/download?id=ea732fb5-4bec-4be7-93f2-8ab91e74a6c6&rid=bf073841-c734-49bf-a97f-3757a6013812"
#url="https://data.tycg.gov.tw/opendata/datalist/datasetMeta/download?id=5ca2bfc7-9ace-4719-88ae-4034b9a5a55c&rid=a1b4714b-3b75-4ff8-a8f2-cc377e4eaa0f"
req = urllib.request.Request(url)
print('here0')
response = urllib.request.urlopen(req, context= context)        #打開HTTPS 網頁連接

if response.code ==200:
    contents = response.read().decode(response.headers.get_content_charset())     #取得網頁資料
    print('here')
    print(contents)
    data = json.loads(contents)      #將JSON純文字資料轉成字典
    print(data["retVal"]["2001"]["sna"])    #顯示公園名稱
    for x in range(2001,2100):
        print(data["retVal"][str(x)]["sna"])
else:
    print('error')
"""


  



