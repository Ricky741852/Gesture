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


from ..models.gesture_detector import GestureDetector

class Image_GroundTruth():
    def __init__(self, index, model_name, Gesture_Data_Model, windows_size=50):
        self.window_size = windows_size
        self.raw_data, self.gesture_label, self.ground_truth, self.gesture_class, self.raw_data_path = Gesture_Data_Model.generate_test_data(index)

        # Raw data plottings
        self.raw_class_list = ['1', '2', '3', '4', '5']
        self.raw_colors = ["orange", "blue", "red", "brown", "green"]

        # 初始化畫布和軸
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(15, 15))

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
        
        self.np_data = np.array(self.raw_data)

        return True

    def generate_static_plot(self):
        # 處理原始數據
        raw_data_T = self.np_data.T
        for i in range(len(self.raw_class_list)):
            self.lines_raw[i].set_ydata(raw_data_T[i])

        # 處理 Ground Truth 數據
        self.lines_truth.set_ydata(self.ground_truth)

        # 儲存靜態圖片
        output_dir = 'output/images/groundtruth'
        os.makedirs(output_dir, exist_ok=True)
        raw_data_filename = self.raw_data_path.split('/')[-1].split('.')[0]

        plt.savefig(os.path.join(output_dir, f'image_groundtruth_{raw_data_filename}.png'))
        plt.show()