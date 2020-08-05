# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 13:09:34 2020

@author: lucifelex
"""

import os
import os.path
import shutil    #複製檔案
import xlrd    #Excell檔案函式褲
import xlwt
import csv

"""
fr = open('workfile.txt','w')       #寫入或創建檔案
fr.write('This is a test\n')
fr.write('This is two test\n')
fr.close()

fw = open('workfile.txt','r')       #讀取檔案
for line in fw:
    print(line)
fw.close()
"""


"""
FileName1 = 'workfile.txt'
FileName2 = 'workfileCopy.txt'
FileName3 = 'workfileRename.txt'

def FunListAllFiles(lineNumber):
    allFiles = os.listdir('.')
    print(lineNumber)
    print(allFiles)

FunListAllFiles("1.")
if os.path.isfile(FileName1) and os.access(FileName1, os.R_OK):     #是否有該檔案
    shutil.copy(FileName1, FileName2)       #複製檔案
    
FunListAllFiles("2.")
if os.path.isfile(FileName2) and os.access(FileName2, os.R_OK):     #是否有該檔案
    os.rename(FileName2, FileName3)         #修改檔名
    
FunListAllFiles('3.')
if os.path.isfile(FileName3) and os.access(FileName3, os.R_OK):     #是否有該檔案
   os.remove(FileName3)     #刪除檔案
"""



"""
if os.path.exists('./folder'):
    os.rmdir('./folder')        #移除資料夾
    print(os.getcwd())
else:
    os.mkdir('./folder')        #建立資料夾
    os.chdir('./folder')        #移動路徑
    print(os.getcwd())
"""    


"""
read = xlrd.open_workbook('workfile.xls')       #打開Excell檔案
sheet = read.sheets()[0]        #第一個Sheet
print(sheet.nrows)      #row筆數
print(sheet.ncols)      #欄位數
write = xlwt.Workbook() #新增一個Excell檔案
write2 = write.add_sheet('MySheet')     #在新增的Excell檔案建立一個Sheet
for i in range(0, sheet.nrows):
    print(sheet.cell(i,1).value)
    value = sheet.cell(i,1).value
    write2.write(i,0,value)     #在新增的Excell檔案,的Sheet裡第i格,第0欄輸入資料
write.save('write.xls')         #以write為名稱,建立excell檔
"""


"""
#CSV
with open('workfile.csv','r') as fin:
    with open('write.csv','w') as fout:
        read = csv.reader(fin, delimiter=',')
        write = csv.writer(fout, delimiter=',')
        header = next(read)     #讀取第一列資料
        print(header)           #第一列,即欄位名稱
        write.writerow(header)  #將表頭欄位寫入列
        for row in read:        #
           print(row)
           print('-------------------')
"""





