import tensorflow as tf
import torch

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from process_6 import Gesture_Data
from test_6 import Gesture_Detection

from collections import namedtuple

# 定義一個名為 PredictData 的 namedtuple，包含 label 和 max_value 兩個屬性
PredictData = namedtuple('PredictData', ['label', 'max_value'])

class Ani_plot():
    def __init__(self, index, model_name, G, windows_size=100):
        self.index = index
        self.model_name = model_name
        self.G = G
        self.window_size = windows_size
        self.raw_datas = None
        # self.predict_data = [0] * (self.window_size - 1)
        # self.predict_data = [PredictData(label=6, max_value=0) for _ in range(self.window_size - 1)]
        self.g_data, self.g_label, self.g_truth, self.g_class_list, self.g_class_name, self.g_data_len = G.generate_test_data(index)

        self.raw_class_list = ['1', '2', '3', '4', '5']
        self.g_class_list = ['1', '2', '3', '4', '5', '6']
        self.plot_color = ["orange", "blue", "red", "brown", "green", "purple"]
        self.predict_data = [PredictData(label=6, max_value=0) for _ in range(self.window_size - 1)]

        self.Gesture_model = Gesture_Detection(model_name, windows_size=windows_size)

        # 初始化畫布和軸
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(15, 15))

        # 畫出原始數據的線條
        # self.lines_raw = [self.ax1.plot([], [], lw=2)[0] for _ in range(5)]
        self.lines_raw = [self.ax1.plot([], [], lw=2, label=f'Sensor {i+1}', color=self.plot_color[i])[0] 
                          for i in range(5)]
        self.ax1.set_xlim(0, self.window_size)
        self.ax1.set_ylim(-100, 100)  # 假設 y 軸範圍在 -200 到 400 之間
        self.ax1.legend(bbox_to_anchor=(0.8, 0.8, 0.3, 0.2), loc='upper right')

        # 畫出預測數據的線條
        # self.lines_predict = [self.ax2.plot([], [], lw=2)[0] for _ in range(6)]
        self.lines_predict = [self.ax2.plot([], [], lw=2, label=f'Gesture {class_num}', color=color)[0]
                              for class_num, color in zip(self.g_class_list, self.plot_color)]
        self.ax2.set_xlim(0, self.window_size)
        self.ax2.set_ylim(0, 1)  # 假設 y 軸範圍在 0 到 1 之間
        self.ax2.legend(bbox_to_anchor=(0.8, 0.8, 0.3, 0.2), loc='upper right')

        self.IoU_text = self.ax2.text(.5, .5, '', fontsize=15)

        # 初始空的視窗
        x = np.arange(0, self.window_size)
        y = np.zeros(self.window_size)
        
        for line in self.lines_raw:
            line.set_data(x, y)

        for line in self.lines_predict:
            line.set_data(x, y)
        

    def generate_data(self):

        if len(self.g_data) < self.window_size:
            print("Data less than {}".format(self.window_size))
            return False
        print("gen_predict_data start")
        
        self.raw_datas = self.g_data
        self.np_data = np.array(self.raw_datas)
        print(self.np_data.shape)
        # exit()

        data = np.array(self.g_data)/360
        self.windows = self.Gesture_model.make_sliding_windows(data)
        predictions = self.Gesture_model.predict(self.windows)

        for i in range(len(predictions)):
            # Check which heatmap index is the max value in
            output = torch.from_numpy(predictions[i])
            max_value = torch.max(output)
            max_value_index = torch.argmax(output)
            predict_label = int((max_value_index % 6) + 1)
            # 使用 PredictData(namedtuple) 替換之前的 namedtuple 物件
            self.predict_data.append(PredictData(label=predict_label, max_value=float(max_value)))
            # self.predict_data.append(float(max_value))

            # print(f'Predict label: {predict_label}, max value: {max_value}')
        return True  # 可根據實際情況返回生成成功或失敗的標誌

    def animate(self, frame):
        if frame + self.window_size <= len(self.raw_datas):
            for i, signal_row in enumerate(np.nditer(self.np_data, flags=['external_loop'], order='F')):
                # 在這裡執行操作，例如將每一列的元素取出
                y_raw = signal_row[frame:frame+self.window_size]
                self.lines_raw[i].set_ydata(y_raw)


            # for i in range(len(self.g_class_list)):
            #     # print(self.predict_data.shape)
            #     y_predict = list(self.predict_data)[frame:frame+self.window_size]
            #     self.lines_predict[i].set_ydata(y_predict)

            # 在這裡使用 namedtuples 的 label 和 max_value 屬性
            y_predict = [data.max_value for data in self.predict_data[frame:frame+self.window_size]]
            # print(y_predict)
            # y_predict = list(self.predict_data)[frame:frame+self.window_size]
            self.lines_predict[0].set_ydata(y_predict)
            # self.IoU_text.set_text(f"predict gesture : {self.predict_data[frame+self.window_size].label}")
            self.IoU_text.set_text(f"predict gesture : {self.predict_data[frame].label}")

        return self.lines_raw, self.lines_predict
        # return self.lines_predict

    def start_animation(self):
        # 建立動畫，設定等待100個sample的時間
        animation = FuncAnimation(self.fig, self.animate, frames=len(self.g_data) - self.window_size, interval=50, repeat=True)

        # 顯示動畫
        plt.rcParams['animation.html'] = 'jshtml'
        plt.show()
        # animation.save('./testAnimation/myAnimation.gif', writer='pillow', fps=20)

if __name__ == "__main__":
    windows_size = 100
    model_name = "model_20231108_100934"

    # G = Gesture_Data(r"./200_new_testData_20231106", windows_size=windows_size)
    G = Gesture_Data(r"./trainData", windows_size=windows_size)
    
    while True:
        index = int(input("-1 to quit : "))
        if index == -1:
            break

        ani = Ani_plot(index, model_name, G, windows_size=windows_size)
        if ani.generate_data():
            ani.start_animation()
