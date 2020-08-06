# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 18:21:17 2020

@author: lucifelex
"""

# 實驗一：使用tf.summary.image紀錄訓練結果

import os
import io
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
#from preprocessing import parse_aug_fn, parse_fn

def parse_aug_fn(dataset):
    """
    Image Augmentation(影像增強) function
    """
    x = tf.cast(dataset['image'], tf.float32) / 255.  # 影像標準化
    x = flip(x)  # 隨機水平翻轉
    # 觸發顏色轉換機率50%
    x = tf.cond(tf.random.uniform([], 0, 1) > 0.5, lambda: color(x), lambda: x)
    # 觸發影像旋轉機率0.25%
    x = tf.cond(tf.random.uniform([], 0, 1) > 0.75, lambda: rotate(x), lambda: x)
    # 觸發影像縮放機率50%
    x = tf.cond(tf.random.uniform([], 0, 1) > 0.5, lambda: zoom(x), lambda: x)
    # 將輸出標籤轉乘One-hot編碼
    y = tf.one_hot(dataset['label'], 10)
    return x, y

def parse_fn(dataset):
    x = tf.cast(dataset['image'], tf.float32) / 255.  # 影像標準化
    # 將輸出標籤轉乘One-hot編碼
    y = tf.one_hot(dataset['label'], 10)
    return x, y


def flip(x):
    """
    flip image(翻轉影像)
    """
    x = tf.image.random_flip_left_right(x)  # 隨機左右翻轉影像
    return x

def color(x):
    """
     Color change(改變顏色)
    """
    x = tf.image.random_hue(x, 0.08)  # 隨機調整影像色調
    x = tf.image.random_saturation(x, 0.6, 1.6)  # 隨機調整影像飽和度
    x = tf.image.random_brightness(x, 0.05)  # 隨機調整影像亮度
    x = tf.image.random_contrast(x, 0.7, 1.3)  # 隨機調整影像對比度
    return x

def rotate(x):
    """
    Rotation image(影像旋轉)
    """
    # 隨機選轉n次(通過minval和maxval設定n的範圍)，每次選轉90度
    x = tf.image.rot90(x,tf.random.uniform(shape=[],minval=1,maxval=4,dtype=tf.int32))
    return x


def zoom(x, scale_min=0.6, scale_max=1.4):
    """
    Zoom Image(影像縮放)
    """
    h, w, c = x.shape
    scale = tf.random.uniform([], scale_min, scale_max)  # 隨機縮放比例
    sh = h * scale  # 縮放後影像長度
    sw = w * scale  # 縮放後影像寬度
    x = tf.image.resize(x, (sh, sw))  # 影像縮放
    x = tf.image.resize_with_crop_or_pad(x, h, w)  # 影像裁減和填補
    return x


# Confusion matrix
# Confusion matrix函數：透過tf.math.confusion_matrix來計算Confusion matrix

y_true = [2, 1, 0, 2, 2, 0, 1, 1]
y_pred = [0, 1, 0, 2, 2, 0, 2, 1]
cm = tf.math.confusion_matrix(y_true, y_pred, num_classes=3).numpy()
print(cm)

# 建立plot_confusion_matrix函式：將剛剛上方計算的Confusion matrix陣列以Matplotlib圖片來表示，而Confusion matrix中的數字改成百分比型式

def plot_confusion_matrix(cm, class_names):
    """
    產生一張圖表示的Confusion matrix
    
    Args:
    cm (shape = [n, n]): 傳入Confusion matrix
    class_names (shape = [n]): 傳入類別
    """
    # 標準化confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
    
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    tick_index = np.arange(len(class_names))
    # matplotlib 3.1.1 bug，如果不設定ylim在[-0.5~2.5]，圖片y軸範圍會被縮小成[0~2]
    plt.ylim([-0.5, 2.5])
    # Y軸顯示類別名稱
    plt.yticks(tick_index, class_names)
    # X軸顯示類別名稱，並將類別名稱旋轉45度(避免文字重疊)
    plt.xticks(tick_index, class_names, rotation=45)
    # 再圖片右邊產生一條顏色刻度條
    plt.colorbar()

    # 在每一格Confusion matrix輸入預測百分比
    threshold = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            # 如果格內背景顏色太深使用白色文字顯示，反之使用黑色文字
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)
            
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    # 將圖片的位置進行調整，避免x或y軸的文字被遮擋
    plt.tight_layout()
    return figure

# Example
img = plot_confusion_matrix(cm, [0, 1, 2])


# 建立plot_to_image函數：將Matplotlib圖片轉TensorFlow型式的圖片，這樣才能夠透過tf.summary.image紀錄影像到TensorBoard

def plot_to_image(figure):
    """將Matplotlib plot的圖片轉TensorFlow的張量格式"""
    # 將Matplotlib plot的圖片以PNG的格式儲存到記憶體中
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # 關閉plt圖片，防止圖片直接顯示在Jupyter notebook介面中
    plt.close(figure)
    buf.seek(0)
    # 將記憶體中的資料轉成TensorFlow格式
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)
    return image

# Example
y_true = [2, 0, 2, 2, 0, 1]
y_pred = [0, 0, 2, 2, 0, 2]
cm = tf.math.confusion_matrix(y_true, y_pred, num_classes=3).numpy()
img = plot_confusion_matrix(cm, [0, 1, 2])
#img_show = plot_to_image(img)



#
# 創建Callback函數
#
# 創建回調函數：訓練過程中每個epoch結束，會產生一張Confusion matrix的圖片，並將圖片紀錄在Tensorboard上

class ConfusionMatrix(tf.keras.callbacks.Callback):
    def __init__(self, log_dir, test_data, class_name):
        super(ConfusionMatrix, self).__init__()
        self.log_dir = log_dir
        self.test_data = test_data
        self.class_names = class_name
        self.num_classes = len(class_name)

    def on_train_begin(self, logs=None):
        path = os.path.join(self.log_dir, 'confusion_matrix')
        # 創建TensorBoard紀錄檔
        self.writer = tf.summary.create_file_writer(path)

    def on_epoch_end(self, epoch, logs=None):
        # 計算Confusion matrix
        total_cm = np.zeros([10, 10])
        for x, y_true in self.test_data:
            y_pred = self.model.predict(x)
            y_pred = np.argmax(y_pred, axis=1)
            y_true = np.argmax(y_true, axis=1)
            cm = tf.math.confusion_matrix(y_true, y_pred, num_classes=self.num_classes).numpy()
            total_cm += cm
        
        # 將Confusion matrix轉成Matplotlib圖片
        figure = plot_confusion_matrix(total_cm, class_names=self.class_names)
        # 將Matplotlib圖片轉成TensorFlow型式的圖片
        cm_image = plot_to_image(figure)

        # 將圖片紀錄在TensorBoard log中
        with self.writer.as_default():
            tf.summary.image("Confusion Matrix", cm_image, step=epoch)
 
    
 
#
# 訓練網路模型
#
# 載入CIFAR-10數據集                        
# 將train Data重新分成9:1等分，分別分給train data, valid data
train_split, valid_split = ['train[:90%]', 'train[90%:]']
# 取得訓練數據，並順便讀取data的資訊
train_data, info = tfds.load("cifar10", split=train_split, with_info=True)
# 取得驗證數據
valid_data = tfds.load("cifar10", split=valid_split)
# 取得測試數據
test_data = tfds.load("cifar10", split=tfds.Split.TEST)
# 取得CIFAR-10數據集的類別
class_name = info.features['label'].names


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
# 建立網路模型
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
logs_dirs = 'lab7-logs-images'
model_cbk = keras.callbacks.TensorBoard(logs_dirs)
# 儲存Confusion matrix圖片
save_cm = ConfusionMatrix(logs_dirs, test_data, class_name)

# 設定訓練使用的優化器、客自化損失函數和客自化指標函數
model_1.compile(keras.optimizers.Adam(), 
                loss=keras.losses.CategoricalCrossentropy(from_logits=True), 
                metrics=[keras.metrics.CategoricalAccuracy()])

# 訓練網路模型

model_1.fit(train_data,
            epochs=3, 
            validation_data=valid_data,
            callbacks=[model_cbk, save_cm])

