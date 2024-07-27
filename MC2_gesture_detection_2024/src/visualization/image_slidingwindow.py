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

from matplotlib.patches import Rectangle

class Image_SlidingWindow():
    def __init__(self, index, model_name, Gesture_Data_Model, windows_size=50):
        self.window_size = windows_size
        self.raw_data, self.gesture_label, self.ground_truth, self.gesture_class, self.raw_data_path = Gesture_Data_Model.generate_test_data(index)

        self.component_len = self.window_size - self.gesture_label[0]
        
        # Raw data plottings
        self.raw_class_list = ['1', '2', '3', '4', '5']
        self.raw_colors = ["orange", "blue", "red", "brown", "green"]

        # 初始化畫布和軸
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(15, 10))

        # 為每個子圖設置 y 軸標籤
        self.ax1.set_ylabel('Raw Data', fontsize=14)
        self.ax2.set_ylabel('Ground Truth', fontsize=14)

        # 畫出原始數據的線條
        self.lines_raw = [self.ax1.plot([], [], lw=2, label=f'Sensor {i+1}', color=self.raw_colors[i])[0] 
                          for i in range(5)]
        self.ax1.set_xlim(0, len(self.raw_data) + self.component_len)
        self.ax1.set_ylim(-110, 110)
        self.ax1.legend(bbox_to_anchor=(0.8, 0.8, 0.3, 0.2), loc='upper right')

        # 原始數據的 Sliding Window
        self.rect = Rectangle((0, -105), self.window_size, 210, facecolor='green', alpha=0.3)
        self.ax1.add_patch(self.rect)
        
        # 畫出Ground Truth數據的線條
        self.lines_truth = self.ax2.plot([], [], lw=2, label=f'Ground Truth')[0]
        self.ax2.set_xlim(0, len(self.raw_data) + self.component_len)
        self.ax2.set_ylim(-0.05, 1.05)
        self.ax2.legend(bbox_to_anchor=(0.8, 0.8, 0.3, 0.2), loc='upper right')

        self.line_truth_x = self.ax2.axvline(x=self.window_size, color='blue', linestyle='--', lw=2)
        self.line_truth_y = self.ax2.axhline(y=0.0, color='blue', linestyle='--', lw=2)

        self.score = self.ax2.text(.5, .5, '', fontsize=15)
        
        # 初始空的視窗
        x = np.arange(0, len(self.raw_data) + self.component_len)
        y = np.zeros(len(x))
        
        for line in self.lines_raw:
            line.set_data(x, y)

        self.lines_truth.set_data(x, y)
        
    def generate_data(self):

        if len(self.raw_data) < self.window_size:
            print("Data less than {}".format(self.window_size))
            return False
        
        self.np_data = np.array(self.raw_data)
        # 最前方補上49個0，以產生第一筆raw data的window，進而對應到 Ground Truth 的第一組分數
        data_with_zeros_front = np.insert(self.np_data, 0, np.zeros((self.component_len, len(self.raw_class_list))), axis=0)
        self.np_data = data_with_zeros_front

        # 在 Ground Truth 前方也補上49個0分，以對應到第一筆raw data的window位置
        self.ground_truth = np.insert(self.ground_truth, 0, np.zeros((self.component_len)), axis=0)

        return True

    def generate_static_plot(self):
        
        frame = 15

        # 處理原始數據
        raw_data_T = self.np_data.T
        for i in range(len(self.raw_class_list)):
            self.lines_raw[i].set_ydata(raw_data_T[i])

        # 處理 Ground Truth 數據
        self.lines_truth.set_ydata(self.ground_truth)

        self.rect.xy = (frame, -105)
        self.line_truth_x.set_xdata(self.window_size + frame)
        self.line_truth_y.set_ydata(self.lines_truth.get_ydata()[self.window_size + frame])
        self.score.set_text(f'Score: {self.lines_truth.get_ydata()[self.window_size + frame]:.2f}')

        # 儲存靜態圖片
        output_dir = 'output/images/slidingwindow'
        os.makedirs(output_dir, exist_ok=True)
        raw_data_filename = self.raw_data_path.split('/')[-1].split('.')[0]

        plt.savefig(os.path.join(output_dir, f'image_slidingwindow_{raw_data_filename}_window_{self.window_size + frame}.png'))
        plt.show()