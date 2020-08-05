# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 16:36:46 2020

@author: lucifelex
"""


# cmd -> python -m http.server 8888    #用命令列方式啟動網頁伺服器

"""
import sys

if (sys.version_info > (3, 0)):    # python 3.x
    import socketserver as socketserver
    import http.server
    from http.server import SimpleHTTPRequestHandler as RequestHandler
else:                              # python 2.x
    import SocketServer as socketserver
    import BaseHTTPServer
    from SimpleHTTPServer import SimpleHTTPRequestHandler as RequestHandler

if sys.argv[1:]:
    port = int(sys.argv[1])
else:
    port = 8888

print('Server listening on port %s' % port)

socketserver.TCPServer.allow_reuse_address = True
httpd = socketserver.TCPServer(('0.0.0.0', port), RequestHandler)
try:
    httpd.serve_forever()
except:
    print("Closing the server.")
    httpd.server_close()
    raise
"""


"""
# socketserver.TCPServer
# RequestHandler
import sys
import time

if (sys.version_info > (3, 0)):    # python 3.x
    import socketserver as socketserver
    import http.server
    from http.server import SimpleHTTPRequestHandler as RequestHandler
else:   # python 2.x
    import SocketServer as socketserver
    import BaseHTTPServer
    from SimpleHTTPServer import SimpleHTTPRequestHandler as RequestHandler

class MyHandler(RequestHandler):        # 繼承 RequestHandler
    def do_GET(self):               # 修改和覆蓋原本HTTP Get方法  
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        print(self.wfile)
        output = b""
        output += b"<html><body>Hello</body></html>"
        self.wfile.write(output)                # 回傳網頁內容給使用者

if sys.argv[1:]:
    port = int(sys.argv[1])
else:
    port = 8888

print('Server listening on port %s' % port)
socketserver.TCPServer.allow_reuse_address = True
httpd = socketserver.TCPServer(('0.0.0.0', port), MyHandler)        # 自訂的 RequestHandler
try:
    httpd.serve_forever()       #啟動網路伺服器
except:
    print("Closing the server.")
    httpd.server_close()
    raise
"""


"""
#  GET 方法處理
import sys
import time

if (sys.version_info > (3, 0)):    # python 3.x
    import socketserver as socketserver
    import http.server
    from http.server import SimpleHTTPRequestHandler as RequestHandler
    from urllib.parse import urlparse
else:   # python 2.x
    import SocketServer as socketserver
    import BaseHTTPServer
    from SimpleHTTPServer import SimpleHTTPRequestHandler as RequestHandler

    from urlparse import urlparse

class MyHandler(RequestHandler):
    def do_HEAD(self):          # 表頭處理
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()

    def do_GET(self):           # 覆蓋原本 HTTP Get 方法
        query = urlparse(self.path).query   # 取得和解析網路完整的URL
        name =b" "
        password =b" "
        if query!="":
           query_components = dict(qc.split("=") for qc in query.split("&"))    取得資料
           name = query_components["name"]
           password = query_components["password"]
        self.do_HEAD()      #呼叫 HTML 表頭處理
        print(self.wfile)
        output = b""
        #output += b"<html><body>Hello name="+b(name) + b" password="+b(password) +b"</body></html>"
        output += b"<html><body>Hello name="
        try:
            output +=name
        except:
            output +=name.encode('utf-8')
        output += b" password="
        try:
            output +=password
        except:
            output +=password.encode('utf-8')
        output += b"</body></html>"
        self.wfile.write(output)        # 回傳網頁內容給使用者
        #self.wfile.close()

if sys.argv[1:]:
    port = int(sys.argv[1])
else:
    port = 8888

print('Server listening on port %s' % port)
socketserver.TCPServer.allow_reuse_address = True
#httpd = socketserver.TCPServer(('127.0.0.1', port), MyHandler)
httpd = socketserver.TCPServer(('0.0.0.0', port), MyHandler)        # 自訂的 RequestHandler
try:
    httpd.serve_forever()
except:
    print("Closing the server.")
    httpd.server_close()
    raise
"""



# POST 方法處理
import sys
import time

from sys import version as python_version
from cgi import parse_header, parse_multipart

if (sys.version_info > (3, 0)):    # python 3.x
    import socketserver as socketserver
    import http.server
    from http.server import SimpleHTTPRequestHandler as RequestHandler
    #from urllib.parse import urlparse
    from urllib.parse import parse_qs
    #from http.server import BaseHTTPRequestHandler


else:   # python 2.x
    import SocketServer as socketserver
    import BaseHTTPServer
    from SimpleHTTPServer import SimpleHTTPRequestHandler as RequestHandler
    #from urlparse import urlparse
    from urlparse import parse_qs

class MyHandler(RequestHandler):
    def do_HEAD(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()

    def do_POST(self):      #自定義 POST 方法
        varLen = int(self.headers['Content-Length'])        #取得使用者傳過來的資料長度
        name =b" "
        password =b" "
        if varLen>0:            #取得和解析完逞的URL
           query_components = parse_qs(self.rfile.read(varLen), keep_blank_values=1)
           print(query_components)
           name = query_components[b"name"][0]
           password = query_components[b"password"][0]
        self.do_HEAD()
        print(self.wfile)
        output = b""
        output += b"<html><body>Hello name="
        output +=name
        output += b" password="
        output +=password
        output += b"</body></html>"
        self.wfile.write(output)       # 回傳網頁內容給使用者
        #self.wfile.close()

if sys.argv[1:]:
    port = int(sys.argv[1])
else:
    port = 8888

print('Server listening on port %s' % port)
socketserver.TCPServer.allow_reuse_address = True
httpd = socketserver.TCPServer(('0.0.0.0', port), MyHandler)

try:
    httpd.serve_forever()
except:
    print("Closing the server.")
    httpd.server_close()
    raise
