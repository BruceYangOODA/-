# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 14:00:41 2020

@author: lucif
"""
import tensorflow as tf 
#import numpy as np 
from tensorflow.examples.tutorials.mnist import input_data 

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

print('start')
#print(mnist.train.images.shape, mnist.train.labels.shape)
#print(mnist.test.images.shape, mnist.test.labels.shape)
#print(mnist.validation.images.shape, mnist.validation.labels.shape)
#result (55000, 784) (55000, 10)
#result (10000, 784) (10000, 10)
#result (5000, 784) (5000, 10)
# mnist.train.labels 是一個10維的向量，只有一個1，其餘是0。位置1的部分代表這張圖代表的數字

# y = softmax(Wx + b)
# W -> weights
# b -> biases
sess = tf.InteractiveSession()
# 建立一個InteractiveSession,並註冊為預設Session，之後運算跑在預設Session內
x = tf.placeholder(tf.float32,[None,784])
# x -> input 第一參數 資料型態；第二參數 資料尺寸(資料筆數),資料資料向量維數
weights = tf.Variable(tf.zeros([784,10]))
# weights -> input 784維的加權指數，輸入784維 輸出10維
biases = tf.Variable(tf.zeros([10]))
# biases -> 
y = tf.nn.softmax(tf.matmul(x,weights) + biases)
# y = softmax(Wx + b)
# Wx -> tf.matmul(x,weights)
# y -> output 一個10維的陣列,內涵之前演算出來的加權比重數值
y_ = tf.placeholder(tf.float32,[None,10])
# y_ -> output 第一參數 資料型態；第二參數 資料尺寸(資料筆數),資料資料向量維數
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
# 實際輸出與訓練答案的平均值設為優化參數
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
# 0.5 學習速率
# cross_entropy 優化目標
tf.global_variables_initializer().run()

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    #在訓練集當中挑出100筆資料訓練
    train_step.run({x: batch_xs, y_ : batch_ys})
    
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
# 預測正確輸出1,預測錯誤輸出0
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# 將correct_prediction轉成0或1，並加總起來算平均數

print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels}))

print('end')

