import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

from matplotlib.animation import FuncAnimation
from ..models.gesture_detector import GestureDetector

class Ani_SlidingWindow():
    def __init__(self, index, model_name, Gesture_Data_Model, windows_size=50):
        self.window_size = windows_size
        self.raw_data, self.gesture_label, self.ground_truth, self.gesture_class, self.raw_data_path = Gesture_Data_Model.generate_test_data(index)

        # Raw data plottings
        self.raw_class_list = ['1', '2', '3', '4', '5']
        self.raw_colors = ["orange", "blue", "red", "brown", "green"]

        # Prediction data plottings
        self.gesture_class_list = ['0', '1', '2', '3']
        self.gesture_colors = ["purple", "orange", "green", "red"]

        self.Gesture_model = GestureDetector(model_name, window_size=windows_size)

        # 初始化畫布和軸
        # self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(15, 15))
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

        # 初始空的視窗
        x = np.arange(0, len(self.raw_data))
        y = np.zeros(len(x))
        
        for line in self.lines_raw:
            line.set_data(x, y)

        self.lines_truth.set_data(x, y)
        
    def generate_data(self):

        if len(self.raw_data) < self.window_size:
            print("Data less than {}".format(self.window_size))
            return False
        print("gen_predict_data start")
        
        self.np_data = np.array(self.raw_data)
        # 最前方補上49個0，以產生第一筆raw data的window，進而預測第一個分數
        data_with_zeros_front = np.insert(self.np_data, 0, np.zeros((self.window_size -1, len(self.raw_class_list))), axis=0)

        data = data_with_zeros_front/360
        self.windows = self.Gesture_model.make_sliding_windows(data)
        self.predict_data = self.Gesture_model.predict(self.windows)

        return True

    def generate_static_plot(self):
        # 處理原始數據
        raw_data_T = self.np_data.T
        for i in range(len(self.raw_class_list)):
            self.lines_raw[i].set_ydata(raw_data_T[i])

        # 處理 Ground Truth 數據
        self.lines_truth.set_ydata(self.ground_truth)

        # 儲存靜態圖片
        output_dir = 'output/images'
        os.makedirs(output_dir, exist_ok=True)
        raw_data_filename = self.raw_data_path.split('/')[-1].split('.')[0]

        plt.savefig(os.path.join(output_dir, f'image_{raw_data_filename}.png'))
        plt.show()

    def start_animation(self):
        # 建立動畫，設定等待100個sample的時間
        animation = FuncAnimation(self.fig, self.animate, frames=len(self.raw_data) - self.window_size + 1, interval=50, repeat=True)

        # 顯示動畫
        plt.rcParams['animation.html'] = 'jshtml'
        plt.show()

        # 儲存動畫
        output_dir = 'output/animations/slidingwindow'
        os.makedirs(output_dir, exist_ok=True)
        raw_data_filename = self.raw_data_path.split('/')[-1].split('.')[0]
        animation.save(os.path.join(output_dir, f'animation_slidingwindow_{raw_data_filename}.gif'), writer='pillow', fps=20)