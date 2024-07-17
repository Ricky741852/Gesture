import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

from matplotlib.animation import FuncAnimation
from collections import deque
from src.models.gesture_detector import GestureDetector

class Ani_Realtime():
    def __init__(self, model_name, SerialUSB, windows_size=50):
        self.window_size = windows_size
        self.SerialUSB = SerialUSB

        # Raw data plottings
        self.raw_class_list = ['1', '2', '3', '4', '5']
        self.raw_colors = ["orange", "blue", "red", "brown", "green"]

        # Prediction data plottings
        self.gesture_class_list = ['0', '1', '2', '3']
        self.gesture_colors = ["purple", "orange", "green", "red"]

        # 載入預測功能模組
        self.Gesture_model = GestureDetector(model_name, window_size=windows_size)

        # 初始化畫布和軸
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(15, 15))

        # 畫出原始數據的線條
        self.lines_raw = [self.ax1.plot([], [], lw=2, label=f'Sensor {raw_num}', color=color)[0] 
                          for raw_num, color in zip(self.raw_class_list, self.raw_colors)]
        self.ax1.set_xlim(0, self.window_size)
        self.ax1.set_ylim(-110, 110)  # 假設 y 軸範圍在 -110 到 110 之間
        # self.ax1.set_ylim(0, 1200)  # 假設 y 軸範圍在 0 到 1200 之間  # 若要畫的是未經Calibration的數據，則使用這行
        self.ax1.legend(bbox_to_anchor=(0.8, 0.8, 0.3, 0.2), loc='upper right')

        # 畫出預測數據的線條
        self.lines_predict = [self.ax2.plot([], [], lw=2, label=f'Gesture {class_num}', color=color)[0]
                              for class_num, color in zip(self.gesture_class_list, self.gesture_colors)]
        self.ax2.set_xlim(0, self.window_size)
        self.ax2.set_ylim(0, 1)  # 假設 y 軸範圍在 0 到 1 之間
        self.ax2.legend(bbox_to_anchor=(0.8, 0.8, 0.3, 0.2), loc='upper right')

        # self.predict_class = self.ax2.text(.5, .5, '', fontsize=15)

        # 初始空的視窗
        x = np.arange(0, self.window_size)
        y = np.zeros(self.window_size)
        
        for line in self.lines_raw:
            line.set_data(x, y)

        for line in self.lines_predict:
            line.set_data(x, y)
    
    def gen_predict_data(self, rawData):
        
        data = np.array(rawData).T/360

        windows = self.Gesture_model.make_sliding_windows(data)
        predictions = self.Gesture_model.predict(windows)

        return predictions
    
    def get_data(self):
        while True:
            raw_data = self.SerialUSB.getSerialData()  # 共有 5 個感測器的數據，每個共有 99 個數據
            predict_data = self.gen_predict_data(raw_data)
            current_raw_data = np.array(raw_data)[:, -50:]
            np_predict_data = np.array(predict_data).T
            yield current_raw_data, np_predict_data

    def animate(self, data):
        raw_data, predict_data = data

        for i in range(len(self.raw_class_list)):
            self.lines_raw[i].set_ydata(raw_data[i])

        for i in range(len(self.gesture_class_list)):
            self.lines_predict[i].set_ydata(predict_data[i])
        
        return tuple(self.lines_raw) + tuple(self.lines_predict)

    def start_animation(self):
        print("start_animation")
        animation = FuncAnimation(self.fig, self.animate, frames=self.get_data, interval=0, blit=True)

        plt.show()
        print("end_animation")