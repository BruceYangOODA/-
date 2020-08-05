# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import tkinter as tk    #視窗GUI
from tkinter import Label, Button, messagebox, Entry, StringVar    #Label Button是在tkinter 模組裡面
from PIL import ImageTk, Image    #PIL 圖片模組 是在tkinter 模組裡面

win = tk.Tk()       #建立GUI應用程式的主視窗
win.wm_title("Tkinter GUI")      #視窗Title
win.minsize(width=666, height=480)
win.maxsize(width=1024, height=1024)
win.resizable(width=False, height=False)

img = ImageTk.PhotoImage(Image.open("python.png"))      #讀取圖片檔
label5 = Label(win, image = img)       #指定Label顯示圖片
label5.place(x=0, y=0)
#label5.pack()

label1 = Label(win,text="Heloow World")     #建立label元件,第一參數是master
label1.pack()       #master加入元件
label2 = Label(win,text="Heloow World",fg="red",bg="yellow")     
label2.pack()       
label3 = Label(win,text="Hellow NO3")
label3.pack(side="top", anchor="w")     #加入元件,靠上方,靠西方 west
label4 = Label(win,text="Hellow No4")
label4.place(x=30, y=100)       # place 指定元件位置 

def iconClick():        # Click 功能
    print("icon Clicked")
icon = ImageTk.PhotoImage(Image.open("icon.jpg"))
btn2 = Button(win,text="press me", image = icon, command = iconClick)
btn2.place(x=480, y=40)

def clickEvent():       # Click功能
    print("btn1 Clicked")
btn1 = Button(win,text = "Click Test", command = clickEvent)
#btn1.place(x=140, y=100)
btn1.pack()
    
def infoEvent():       # MessageBox Infor功能
    messagebox.showinfo("Info Title", "showinfo")    
    
def errorEvent():       # MessageBox Error功能
    messagebox.showerror("Error Title", "showerror")
    
def warningEvent():     # MessageBox Waring功能
    messagebox.showwarning("Waring Title", "showwaring")
    
def questionEvent():     # MessageBox Question功能
    result = messagebox.askquestion("Question Title", "askquestion")
    print(result)       # yes no

def askOkCancelEvent():     # MessageBox Ok Cancel功能
    result = messagebox.askokcancel("Ok Cancel Title", "askokcancel")
    print(result)       # True False

def askYesNoEvent():     # MessageBox Yes No功能
    result = messagebox.askyesno("YesNo Title", "askyesno")
    print(result)       # True False

def askRetryCancelEvent():     # MessageBox Retry Cancel功能
    result = messagebox.askretrycancel("Retry Title", "askretrycancel")
    print(result)       # True False

btnInfo = Button(win, text = "Show Info", command = infoEvent)
btnInfo.place(x=140, y=100)

btnError = Button(win, text = "Show Error", command = errorEvent)
btnError.place(x= 30 , y= 140)

btnWaring = Button(win, text = "Show Waring", command = warningEvent)
btnWaring.place(x= 140, y = 140)

btnQuestion = Button(win, text ="askquestion", command = questionEvent)
btnQuestion.place(x= 30, y = 180)

btnAskOkCancel = Button(win, text = "askokcancel", command = askOkCancelEvent)
btnAskOkCancel.place(x=140, y= 180)

btnAskYesNo = Button(win, text = "askyesno", command = askYesNoEvent)
btnAskYesNo.place(x=30, y= 220)

btnRetryCancel = Button(win, text = "askretrycancel", command = askRetryCancelEvent)
btnRetryCancel.place(x= 140, y = 220)

entryInput = Entry(win)     #建立輸入框
entryInput.pack()       #在視窗放置輸入框
textVariable = StringVar()      #文字變數
labelShowEntry = Label(win, text="Hello", textvariable = textVariable)      #設定Label 文字變數
labelShowEntry.pack()
textVariable.set("Text Variable")       #文字變數 設定其值

def entryPrintEnent():      #將輸入框輸入設給文字變數, Label跟著文字變數修改
    print(entryInput.get())
    textInput = entryInput.get()
    textVariable.set(textInput)
btnEntry = Button(win, text="Entry Enter", command = entryPrintEnent)
btnEntry.place(x=410, y= 90)


win.mainloop()      #程式做無限循環


