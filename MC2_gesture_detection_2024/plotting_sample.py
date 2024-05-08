import tensorflow as tf
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

import numpy as np
import matplotlib.pyplot as plt
from animation_process import Gesture_Data
from test import Gesture_Detection

from collections import namedtuple

# 定義一個名為 PredictData 的 namedtuple，包含 label 和 max_value 兩個屬性
PredictData = namedtuple('PredictData', ['label', 'max_value'])

class Plot():
    def __init__(self, index, model_name, Gesture_Data_Model, windows_size=50):
        self.index = index
        self.model_name = model_name
        self.window_size = windows_size
        self.raw_data, self.gesture_label, self.ground_truth, self.gesture_class_list, self.g_class_name = Gesture_Data_Model.generate_test_data(index)

        # Raw data plottings
        self.raw_class_list = ['1', '2', '3', '4', '5']
        self.raw_colors = ["orange", "blue", "red", "brown", "green"]

        # Prediction data plottings
        self.g_colors = ["orange", "blue", "red", "brown", "green", "purple"]
        # 取得所有分類的最高分數
        self.predict_data = [PredictData(label=6, max_value=0) for _ in range(self.window_size - 1)]
        # 取得每個分類的最高分數
        self.predict_data_per_category = [list() for _ in range(len(self.gesture_class_list))]

        self.Gesture_model = Gesture_Detection(model_name, windows_size=windows_size)

        # 初始化畫布和軸
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(3, 1, figsize=(15, 15))

        # 畫出原始數據的線條
        self.lines_raw = [self.ax1.plot([], [], lw=2, label=f'Sensor {i+1}', color=self.raw_colors[i])[0] 
                          for i in range(5)]
        self.ax1.set_xlim(0, len(self.raw_data))
        self.ax1.set_ylim(-110, 110)  # 假設 y 軸範圍在 -200 到 400 之間
        self.ax1.legend(bbox_to_anchor=(0.8, 0.8, 0.3, 0.2), loc='upper right')

        # 畫出Ground Truth數據的線條
        self.lines_truth = self.ax2.plot([], [], lw=2, label=f'Ground Truth')[0]
        self.ax2.set_xlim(0, len(self.raw_data))
        self.ax2.set_ylim(0, 1)
        self.ax2.legend(bbox_to_anchor=(0.8, 0.8, 0.3, 0.2), loc='upper right')

        # 畫出預測數據的線條
        self.lines_predict = [self.ax3.plot([], [], lw=2, label=f'Gesture {class_num}', color=color)[0]
                              for class_num, color in zip(self.gesture_class_list, self.g_colors)]
        self.ax3.set_xlim(0, len(self.raw_data))
        self.ax3.set_ylim(0, 1)  # 假設 y 軸範圍在 0 到 1 之間
        self.ax3.legend(bbox_to_anchor=(0.8, 0.8, 0.3, 0.2), loc='upper right')

        # 初始空的視窗
        x = np.arange(0, len(self.raw_data))
        y = np.zeros(len(x))
        
        for line in self.lines_raw:
            line.set_data(x, y)

        self.lines_truth.set_data(x, y)

        for line in self.lines_predict:
            line.set_data(x, y)
        
    def generate_data(self):

        if len(self.raw_data) < self.window_size:
            print("Data less than {}".format(self.window_size))
            return False
        print("gen_predict_data start")
        
        self.raw_data = self.raw_data
        self.np_data = np.array(self.raw_data)
        data_with_zeros_front = np.insert(self.np_data, 0, np.zeros((self.window_size -1, len(self.raw_class_list))), axis=0)
        self.np_data = data_with_zeros_front

        data = self.np_data/360
        self.windows = self.Gesture_model.make_sliding_windows(data)
        predictions = self.Gesture_model.predict(self.windows)

        for i in range(len(predictions)):
            for j in range(6):
                self.predict_data_per_category[j].append((predictions[i][j]))
        return True

    def generate_static_plot(self):
            # 處理原始數據
            for i, signal_row in enumerate(np.nditer(self.np_data, flags=['external_loop'], order='F')):
                y_raw = signal_row[windows_size - 1:]
                self.lines_raw[i].set_ydata(y_raw)

            # # 處理 Ground Truth 數據
            y_truth = self.ground_truth
            y_truth[y_truth == 0] = np.nan # 將 0 的值轉換為 NaN，這樣就不會在圖上畫出來
            self.lines_truth.set_ydata(y_truth)

            # # 處理預測數據
            for i in range(len(self.gesture_class_list)):
                y_predict = self.predict_data_per_category[i]
                self.lines_predict[i].set_ydata(y_predict)

            # 儲存靜態圖片
            plt.savefig('./static_plot/myStaticPlot.png')
            plt.show()

if __name__ == "__main__":
    windows_size = 50
    model_name = "model_20240401_011610"

    G = Gesture_Data(r"./trainData", windows_size=windows_size)
    
    while True:
        index = int(input("-1 to quit : "))
        if index == -1:
            break

        ani = Plot(index, model_name, G, windows_size=windows_size)
        if ani.generate_data():
            ani.generate_static_plot()
