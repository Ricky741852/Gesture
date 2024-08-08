import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.gridspec import GridSpec

os.environ['KMP_DUPLICATE_LIB_OK']='True'

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


from ..models.gesture_detector import GestureDetector

class Image_RawData():
    def __init__(self, index, model_name, Gesture_Data_Model, windows_size=50):
        self.window_size = windows_size
        self.raw_data, self.gesture_label, self.ground_truth, self.gesture_class, self.raw_data_path = Gesture_Data_Model.generate_test_data(index)

        # Raw data plottings
        self.raw_class_list = ['1', '2', '3', '4', '5']
        self.raw_colors = ["orange", "blue", "red", "brown", "green"]

        # 使用 GridSpec 來手動調整子圖布局
        self.fig = plt.figure(figsize=(15, 5))
        gs = GridSpec(1, 1, height_ratios=[1])

        self.ax1 = self.fig.add_subplot(gs[0, 0])

        # 為每個子圖設置 y 軸標籤
        self.ax1.set_ylabel('Raw Data', fontsize=25)

        # 畫出原始數據的線條
        self.lines_raw = [self.ax1.plot([], [], lw=2, label=f'Sensor {i+1}', color=self.raw_colors[i])[0] 
                          for i in range(5)]
        self.ax1.set_xlim(0, len(self.raw_data))
        self.ax1.set_ylim(-110, 110)

        # 放置圖例
        self.ax1.legend(loc='lower left', bbox_to_anchor=(1, 0.4), fontsize=20)

        # 初始空的視窗
        x = np.arange(0, len(self.raw_data))
        y = np.zeros(len(x))
        
        for line in self.lines_raw:
            line.set_data(x, y)

        # 調整子圖布局
        plt.tight_layout(rect=[0, 0, 1, 1])
        
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

        # 儲存靜態圖片
        output_dir = 'output/images/raw_data'
        os.makedirs(output_dir, exist_ok=True)
        raw_data_filename = self.raw_data_path.split('/')[-1].split('.')[0]

        plt.savefig(os.path.join(output_dir, f'image_rawdata_{raw_data_filename}.png'))
        plt.show()