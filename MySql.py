# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 16:15:52 2020

@author: lucifelex
"""

import pymysql 

db = pymysql.connect(host="127.0.0.1", user="admin",passwd="admin",db = "mydatabase")
cursor = db.cursor()



"""
# INSERT
sql = "insert into mytable (value01,value02,value03,value04) values ('1','1','1','1');"
cursor.execute(sql)
db.commit()

sql = "select * from mytable"
cursor.execute(sql)
result = cursor.fetchall()
for record in result:
    print("value01=%s value02=%s" %(record[0],record[1]))
"""



"""
# UPDATE
#sql = "UPDATE `mytable` SET `value01`='2' WHERE `value01`='1'"
sql = "update `mytable` set `value01`='2' where `value01`='1';"
cursor.execute(sql)
db.commit()
"""



"""
# DELETE
sql = "delete from `mytable` where `value01`='2';"
cursor.execute(sql)
db.commit()
"""






