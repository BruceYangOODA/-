# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 11:47:35 2020

@author: lucifelex
"""





import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

"""
tf.summary
tf.summary是TensorFlow提供TensorBoard低階API指令，主要是用來紀錄log檔。 以下整理了幾個常用功能：

tf.scalar：儲存顯示如損失、指標或學習率等的變化趨勢。
tf.image：儲存顯示影像。
tf.audio：儲存顯示可播放的音頻。
tf.histogram：儲存顯示模型權重。
tf.text：儲存顯示一段文字。
"""
# 創建TensorBoard log檔
summary_writer = tf.summary.create_file_writer('lab7-logs-summary')

# 在0~2π之間產生100個點
x = np.linspace(0, 2 * np.pi , 100)
# 將100個點帶入sin函數中
data = np.sin(x)
with summary_writer.as_default():  # summary_writer作為預設寫入的紀錄檔
    for i, y in enumerate(data):
        tf.summary.scalar('sin', y, step=i)  # 存入數值(y為數值，i為時間軸)
        
    
#
# Image
#
# 儲存一張影像在紀錄檔中並顯示

# 建立讀取影像的函數
def read_img(file):
    image_string = tf.io.read_file(file)  # 讀取檔案
    # 將讀入檔案以影像格式來解碼
    image_decode = tf.image.decode_image(image_string)
    # 將影像增加一個維度(number,height,width,channel)方便之後存入紀錄檔中
    # image_decode = tf.expand_dims(image_decode, axis=0)
    return image_decode

img = read_img('image/airplane.png')  # 讀入影像資訊
plt.imshow(img)  # 顯示讀入的影像資訊


image_string = tf.io.read_file('image/airplane.png')  # 讀取檔案
image_decode = tf.image.decode_image(image_string)

with summary_writer.as_default():  # summary_writer作為預設寫入的紀錄檔
    tf.summary.image("Airplane", [image_decode], step=0)  # 存入影像資訊


img_files = ['airplane_zoom.png', 'airplane_flip.png', 'airplane_color.png', 'airplane_rot.png',
             'airplane.png']  # 創建一個陣列用來儲存讀入的影像 

imgs = [] 

for file in img_files: 
    imgs.append(read_img('image/'+file))  # 讀取影像並存入陣列中 

with summary_writer.as_default():  # summary_writer作為預設寫入的紀錄檔 
    # 一次存入五張影像(注意:如果max_outputs沒設定為5，就只會儲存3張影像) 
    tf.summary.image("Airplane Augmentation", imgs, max_outputs=5, step=0)
    

# 將五張影像以不同Step(時間)儲存

with summary_writer.as_default():  # summary_writer作為預設寫入的紀錄檔
    # 每次儲存一張影像，並儲存在不同Step中
    for i, img in enumerate(imgs):
        tf.summary.image("Save image each step", [img], step=i)


#
# Text
#
# 建立一個陣列，裡面包含了對話記錄
texts = ["小明：Cubee小助理最近好想學深度學習的技術哦!", 
         "Cubee：這是當然的阿，這可現今最火的技術呢!", 
         "小明：那我該如何入門呢?", 
         "Cubee：推薦你一本書「輕鬆學會Google TensorFlow2.0深度學習」。", 
         "小明：這本書沒有深度學習經驗的人也能學會嗎?", 
         "Cubee：這是當然的，你只需要基礎Python能力就可以學會了!", 
         "小明：太好了那我要趕快去買了!"]

with summary_writer.as_default():  # summary_writer作為預設寫入的紀錄檔
    # 將每一段字串資訊以不同Step存入到記錄檔中
    for i, text in enumerate(texts):
        tf.summary.text("Chat record", text, step=i)


# 
# Audio
#
# 建立讀取音訊的函數
def read_audio(file):
    audio_string = tf.io.read_file(file)  # 讀取檔案
    # 將讀入檔案以音訊格式來解碼
    audio, fs = tf.audio.decode_wav(audio_string)
    # 因為tf.summary.audio要求輸入格式為[k(clips), t(frames), c(channels)]
    # 而解碼後的音訊只有[t(frames), c(channels)]，所以需要增加一個維度給音訊
    audio = tf.expand_dims(audio, axis=0)
    return audio, fs

audio, fs = read_audio('./audio/cat.wav')  # 讀取音訊檔

with summary_writer.as_default():  # summary_writer作為預設寫入的紀錄檔
    tf.summary.audio('cat', audio, fs, step=0)  # 存入音訊資訊



"""
Histogram
目前 TensorFlow-gpu 2.0-alpha有問題，等待下一版修復。

https://github.com/tensorflow/tensorboard/issues/1993
"""

# 建立一個常態分佈
data = tf.random.normal([64, 100], dtype=tf.float64)
# 儲存常態分佈分佈
with summary_writer.as_default():
    tf.summary.histogram('Normal distribution', data, step=0)
# 儲存多個常態分佈，並且各個之間平均值都相差0.01
with summary_writer.as_default():
    for i, offset in enumerate(tf.range(0, 10, delta=0.1, dtype=tf.float64)):
        tf.summary.histogram('Normal distribution 2', data+offset, step=i)







        
