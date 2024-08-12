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

from src.data import GestureDataHandler

class Image_GroundTruth():
    def __init__(self, index, datasets_dir, window_size=50):
        # 取得測試資料
        data_handler = GestureDataHandler(datasets_dir, window_size=window_size)

        self.window_size = window_size
        self.raw_data, self.gesture_label, self.ground_truth, self.gesture_class, self.raw_data_path = data_handler.generate_test_data(index)

        # Raw data plottings
        self.raw_class_list = ['1', '2', '3', '4', '5']
        self.raw_colors = ["orange", "blue", "red", "brown", "green"]

        # Gesture data plottings
        self.gesture_class_list = ['0', '1', '2', '3']
        self.gesture_colors = ["purple", "orange", "green", "red"]

        # 每個分類的真實分數
        self.ground_truths = [0 for _ in range(len(self.gesture_class_list))]

        # 使用 GridSpec 來手動調整子圖布局
        self.fig = plt.figure(figsize=(15, 10))
        gs = GridSpec(2, 1, height_ratios=[1, 1])

        self.ax1 = self.fig.add_subplot(gs[0, 0])
        self.ax2 = self.fig.add_subplot(gs[1, 0])

        # 為每個子圖設置 y 軸標籤
        self.ax1.set_ylabel('Raw Data', fontsize=25)
        self.ax2.set_ylabel('Ground Truth', fontsize=25)

        # 畫出原始數據的線條
        self.lines_raw = [self.ax1.plot([], [], lw=2, label=f'Sensor {i+1}', color=self.raw_colors[i])[0] 
                          for i in range(5)]
        self.ax1.set_xlim(0, len(self.raw_data))
        self.ax1.set_ylim(-110, 110)
        self.ax1.legend(loc='lower left', bbox_to_anchor=(1, 0.4), fontsize=20)

        # 畫出Ground Truth數據的線條
        self.lines_truth = [self.ax2.plot([], [], lw=2, label=f'Gesture {class_num}', color=color)[0]
                              for class_num, color in zip(self.gesture_class_list, self.gesture_colors)]
        self.ax2.set_xlim(0, len(self.raw_data))
        self.ax2.set_ylim(-0.05, 1.05)
        self.ax2.legend(loc='lower left', bbox_to_anchor=(1, 0.5), fontsize=20)

        # 初始空的視窗
        x = np.arange(0, len(self.raw_data))
        y = np.zeros(len(x))
        
        for line in self.lines_raw:
            line.set_data(x, y)

        for line in self.lines_truth:
            line.set_data(x, y)

        # 調整子圖布局
        plt.tight_layout(rect=[0, 0, 1, 1])
        
    def generate_data(self):

        if len(self.raw_data) < self.window_size:
            print("Data less than {}".format(self.window_size))
            return False
        
        self.np_data = np.array(self.raw_data)

        self.ground_truths[self.gesture_class_list.index(self.gesture_class)] = self.ground_truth
        # 將背景手勢的分數放入對應的分類中
        self.ground_truths[0] = [1 - gt for gt in self.ground_truth]

        return True

    def generate_static_plot(self):
        # 處理原始數據
        raw_data_T = self.np_data.T
        for i in range(len(self.raw_class_list)):
            self.lines_raw[i].set_ydata(raw_data_T[i])

        # 處理 Ground Truth 數據
        for i in range(len(self.gesture_class_list)):
            self.lines_truth[i].set_ydata(self.ground_truths[i])

        # 儲存靜態圖片
        output_dir = 'output/images/groundtruth'
        os.makedirs(output_dir, exist_ok=True)
        raw_data_filename = self.raw_data_path.split('/')[-1].split('.')[0]

        plt.savefig(os.path.join(output_dir, f'image_groundtruth_{raw_data_filename}.png'))
        plt.show()

def image_groundtruth(index, datasets_dir, window_size=50):
    """
    Plot the image of the ground truth based on the given index.

    Parameters:
    - index (int): The index of the file to be plotted.
    - model_name (str): The name of the model.
    - datasets_dir (str): The directory of the datasets.
    - window_size (int): The window size for the data. Default is 50.

    Returns:
    None
    """
    img = Image_GroundTruth(index, datasets_dir, window_size=window_size)
    if img.generate_data():
        img.generate_static_plot()