# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 10:54:21 2020

@author: lucifelex
"""
"""
Custom Layers
透過繼承tf.keras.layers.Layer類別，來輕鬆創建字定義的網路層。

你可以到https://www.tensorflow.org/api_docs/python/tf/keras/layers 官方API察看更多的網路層。

Example: 簡單的客自化Convolutional layers。
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import initializers
import sys
sys.path.append('C:\TestAi\TensorFlow2.0\Lab6')
import preprocessing
#import preprocessing.parse_fn
#from preprocessing import parse_aug_fn, parse_fn


class CustomConv2D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides=(1, 1), padding="VALID", **kwargs):
        super(CustomConv2D, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = (1, *strides, 1)
        self.padding = padding

    def build(self, input_shape):
        kernel_h, kernel_w = self.kernel_size
        input_dim = input_shape[-1]
        # 創建卷積的權重值(weights)
        self.w = self.add_weight(name='kernel', 
                                 shape=(kernel_h, kernel_w, input_dim, self.filters),
                                 initializer='glorot_uniform',  # 設定初始化方法
                                 trainable=True)  # 設定這個權重是否能夠訓練(更動)
        # 創建卷積的偏差值(bias)
        self.b = self.add_weight(name='bias', 
                                 shape=(self.filters,),
                                 initializer='zeros',  # 設定初始化方法
                                 trainable=True)  # 設定這個權重是否能夠訓練(更動)

    def call(self, inputs):
        x = tf.nn.conv2d(inputs, self.w, self.strides, padding=self.padding) # 卷積運算
        x = tf.nn.bias_add(x, self.b)  # 加上偏差值
        x = tf.nn.relu(x)  # 激活函數
        return x


# Example: 簡單的客自化Crossentropy Loss。

def custom_categorical_crossentropy(y_true, y_pred):
    # x = tf.reduce_mean(-tf.reduce_sum(y_true * tf.log(y_pred), reduction_indices=[1]))
    x = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
    return x


# Example: 計算多少個樣本是正確分類的
class CustomCategoricalAccuracy(tf.keras.metrics.Metric):
    def __init__(self, name='custom_catrgorical_accuracy', **kwargs):
        super(CustomCategoricalAccuracy, self).__init__(name=name, **kwargs)
        # 記錄正確預測的數量
        self.correct = self.add_weight('correct_numbers', initializer='zeros')
        # 記錄全部資料的量數
        self.total = self.add_weight('total_numbers', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        # 輸入答案為One-hot編碼，所以取最大的數值為答案
        y_true = tf.argmax(y_true, axis=-1)
        # 取預測輸出最大的數值為預測結果
        y_pred = tf.argmax(y_pred, axis=-1)
        # 比較預測結果是否正確，正確會返回True(正確)，錯誤會返回False(錯誤)
        values = tf.equal(y_true, y_pred)
        # 轉為浮點數True(正確)=1.0，False(錯誤)=0.0
        values = tf.cast(values, tf.float32)
        # 將values所有數值相加就會等於正確預測的總數
        values_sum = tf.reduce_sum(values)
        # 計算這個Batch的資料數量
        num_values = tf.cast(tf.size(values), tf.float32)
        self.correct.assign_add(values_sum)  # 更新正確預測的總數
        self.total.assign_add(num_values)  # 更新資料量的總數

    def result(self):
        # 計算準確率
        return tf.math.divide_no_nan(self.correct, self.total)

    def reset_states(self):
        # 每一次Epoch結束後會重新初始化變數
        self.correct.assign(0.)
        self.total.assign(0.)



# Example: 紀錄每一個batch的loss值。
class SaveModel(tf.keras.callbacks.Callback):
    def __init__(self, weights_file, monitor='loss', mode='min', save_weights_only=False):
        super(SaveModel, self).__init__()
        self.weights_file = weights_file
        self.monitor = monitor
        self.mode = mode
        self.save_weights_only = save_weights_only
        if mode == 'min':
            self.best = np.Inf
        else:
            self.best = -np.Inf
        
    def save_model(self):
        if self.save_weights_only:
            self.model.save_weights(self.weights_file)
        else:
            self.model.save(self.weights_file)

    def on_epoch_end(self, epoch, logs=None):
        monitor_value = logs.get(self.monitor)
        if self.mode == 'min' and monitor_value < self.best:
            self.save_model()
            self.best = monitor_value
        elif self.mode == 'max' and monitor_value > self.best:
            self.save_model()
            self.best = monitor_value



#
# 載入Cifar10數據集
#
# 將train Data重新分成9:1等分，分別分給train data, valid data
train_split, valid_split = ['train[:90%]', 'train[90%:]']
# 取得訓練數據，並順便讀取data的資訊
train_data, info = tfds.load("cifar10", split=train_split, with_info=True)
# 取得驗證數據
valid_data = tfds.load("cifar10", split=valid_split)
# 取得測試數據
test_data = tfds.load("cifar10", split=tfds.Split.TEST)




#
# Dataset 設定
#
AUTOTUNE = tf.data.experimental.AUTOTUNE  # 自動調整模式
batch_size = 64  # 批次大小
train_num = int(info.splits['train'].num_examples / 10) * 9  # 訓練資料數量

train_data = train_data.shuffle(train_num)  # 打散資料集
# 載入預處理「 parse_aug_fn」function，cpu數量為自動調整模式
train_data = train_data.map(map_func=parse_aug_fn, num_parallel_calls=AUTOTUNE)
# 設定批次大小並將prefetch模式開啟(暫存空間為自動調整模式)
train_data = train_data.batch(batch_size).prefetch(buffer_size=AUTOTUNE)

# 載入預處理「 parse_fn」function，cpu數量為自動調整模式
valid_data = valid_data.map(map_func=parse_fn, num_parallel_calls=AUTOTUNE)
# 設定批次大小並將prefetch模式開啟(暫存空間為自動調整模式)
valid_data = valid_data.batch(batch_size).prefetch(buffer_size=AUTOTUNE)

# 載入預處理「 parse_fn」function，cpu數量為自動調整模式
test_data = test_data.map(map_func=parse_fn, num_parallel_calls=AUTOTUNE)
# 設定批次大小並將prefetch模式開啟(暫存空間為自動調整模式)
test_data = test_data.batch(batch_size).prefetch(buffer_size=AUTOTUNE)



#
# 使用Keras高階API訓練網路模型
#
inputs = keras.Input(shape=(32, 32, 3))
x = layers.Conv2D(64, 3, activation='relu', kernel_initializer='glorot_uniform')(inputs)
x = layers.MaxPool2D()(x)
x = layers.Conv2D(128, 3, activation='relu', kernel_initializer='glorot_uniform')(x)
x = layers.Conv2D(256, 3, activation='relu', kernel_initializer='glorot_uniform')(x)
x = layers.Conv2D(128, 3, activation='relu', kernel_initializer='glorot_uniform')(x)
x = layers.Conv2D(64, 3, activation='relu', kernel_initializer='glorot_uniform')(x)
x = layers.Flatten()(x)
x = layers.Dense(64, activation='relu')(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(10)(x)
# 建立網路模型(將輸入到輸出所有經過的網路層連接起來)
model_1 = keras.Model(inputs, outputs, name='model-1')
model_1.summary()


#
# 建立Callback function
#
# 儲存訓練記錄檔
logs_dirs = 'lab6-logs'
model_cbk = keras.callbacks.TensorBoard(log_dir='lab6-logs')

# 紀錄每一個batch的Loss值變化
model_dirs = logs_dirs + '/models'
os.makedirs(model_dirs, exist_ok=True)
save_model = tf.keras.callbacks.ModelCheckpoint(model_dirs + '/save.h5', 
                                                monitor='val_catrgorical_accuracy', 
                                                mode='max')

# 設定訓練使用的優化器、客自化損失函數和客自化指標函數
# 設定訓練使用的優化器、損失函數和指標函數
model_1.compile(keras.optimizers.Adam(), 
                loss=keras.losses.CategoricalCrossentropy(from_logits=True), 
                metrics=[keras.metrics.CategoricalAccuracy()])

# 訓練網路模型
model_1.fit(train_data,
            epochs=100, 
            validation_data=valid_data,
            callbacks=[model_cbk, save_model])

"""
使用客自化API訓練網路模型
Custom Layer：將原本的Conv2d改成CustomConv2D。
Custom Loss：將原本的CategoricalCrossentropy改成custom_loss。
Custom Metrics：加入一個新的指標函數CatgoricalTruePositives。
Custom Callbacks：紀錄每一個batch的Loss值變化。
"""
inputs = keras.Input(shape=(32, 32, 3))
x = CustomConv2D(64, (3, 3))(inputs)
x = layers.MaxPool2D()(x)
x = CustomConv2D(128, (3, 3))(x)
x = CustomConv2D(256, (3, 3))(x)
x = CustomConv2D(128, (3, 3))(x)
x = CustomConv2D(64, (3, 3))(x)
x = layers.Flatten()(x)
x = layers.Dense(64, activation='relu')(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(10)(x)
# 建立網路模型(將輸入到輸出所有經過的網路層連接起來)
model_2 = keras.Model(inputs, outputs, name='model-2')
model_2.summary()


# 儲存訓練記錄檔
logs_dirs = 'lab6-logs'
model_cbk = keras.callbacks.TensorBoard(log_dir='lab6-logs')

# 紀錄每一個batch的Loss值變化
model_dirs = logs_dirs + '/models'
os.makedirs(model_dirs, exist_ok=True)
custom_save_model = SaveModel(model_dirs + '/custom_save.h5', 
                              monitor='val_custom_catrgorical_accuracy', 
                              mode='max', 
                              save_weights_only=True)

# 設定訓練使用的優化器、損失函數和指標函數
model_2.compile(keras.optimizers.Adam(), 
           loss=custom_categorical_crossentropy, 
           metrics=[CustomCategoricalAccuracy()])

# 訓練網路模型
model_2.fit(train_data,
            epochs=100, 
            validation_data=valid_data,
            callbacks=[model_cbk, custom_save_model])



# 比較兩種方法的訓練結果
# 載入兩種方法的模型權重：
model_1.load_weights(model_dirs+'/save.h5')
model_2.load_weights(model_dirs+'/custom_save.h5')
#驗證網路模型：

loss_1, acc_1 = model_1.evaluate(test_data)
loss_2, acc_2 = model_2.evaluate(test_data)
loss = [loss_1, loss_2]  
acc = [acc_1, acc_2]
dict = {"Loss": loss, "Accuracy": acc}
pd.DataFrame(dict)


