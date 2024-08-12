import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle

os.environ['KMP_DUPLICATE_LIB_OK']='True'

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)



from src.data import GestureDataHandler

class Ani_SlidingWindow():
    def __init__(self, index, datasets_dir, window_size=50):
        # 取得測試資料
        data_handler = GestureDataHandler(datasets_dir, window_size=window_size)

        self.window_size = window_size
        self.raw_data, self.gesture_label, self.ground_truth, self.gesture_class, self.raw_data_path = data_handler.generate_test_data(index)

        self.component_len = self.window_size - self.gesture_label[0]

        # Raw data plottings
        self.raw_class_list = ['1', '2', '3', '4', '5']
        self.raw_colors = ["orange", "blue", "red", "brown", "green"]

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
        self.ax1.set_xlim(0, len(self.raw_data) + self.component_len)
        self.ax1.set_ylim(-110, 110)
        self.ax1.legend(loc='lower left', bbox_to_anchor=(1, 0.4), fontsize=20)

        # 原始數據的 Sliding Window
        self.rect = Rectangle((0, -105), self.window_size, 210, facecolor='green', alpha=0.3)
        self.ax1.add_patch(self.rect)

        # 畫出Ground Truth數據的線條
        self.lines_truth = self.ax2.plot([], [], lw=2, label=f'Ground Truth')[0]
        self.ax2.set_xlim(0, len(self.raw_data) + self.component_len)
        self.ax2.set_ylim(-0.05, 1.05)
        self.ax2.legend(loc='lower left', bbox_to_anchor=(1, 0.5), fontsize=20)

        self.line_truth_x = self.ax2.axvline(x=self.window_size, color='blue', linestyle='--', lw=2)
        self.line_truth_y = self.ax2.axhline(y=0.0, color='blue', linestyle='--', lw=2)

        self.score = self.ax2.text(.5, .5, '', fontsize=15)

        # 初始空的視窗
        x = np.arange(0, len(self.raw_data) + self.component_len)
        y = np.zeros(len(x))
        
        for line in self.lines_raw:
            line.set_data(x, y)

        self.lines_truth.set_data(x, y)

        # 調整子圖布局
        plt.tight_layout(rect=[0, 0, 1, 1])
        
    def generate_data(self):

        if len(self.raw_data) < self.window_size:
            print("Data less than {}".format(self.window_size))
            return False        
        
        self.np_data = np.array(self.raw_data)
        # 最前方補上49個0，以產生第一筆raw data的window，進而對應到 Ground Truth 的第一組分數
        self.np_data = np.insert(self.np_data, 0, np.zeros((self.component_len, len(self.raw_class_list))), axis=0)

        # 在 Ground Truth 前方也補上49個0分，以對應到第一筆raw data的window位置
        self.ground_truth = np.insert(self.ground_truth, 0, np.zeros((self.component_len)), axis=0)

        return True
    
    def generate_static_plot(self):
        # 處理原始數據
        raw_data_T = self.np_data.T
        for i in range(len(self.raw_class_list)):
            self.lines_raw[i].set_ydata(raw_data_T[i])

        # 處理 Ground Truth 數據
        self.lines_truth.set_ydata(self.ground_truth)

        return True

    def animate(self, frame):
        self.rect.xy = (frame, -105)
        self.line_truth_x.set_xdata(self.window_size + frame)
        self.line_truth_y.set_ydata(self.lines_truth.get_ydata()[self.window_size + frame])
        self.score.set_text(f'Score: {self.lines_truth.get_ydata()[self.window_size + frame]:.2f}')
    
        return self.rect, self.line_truth_x, self.line_truth_y, self.score

    def start_animation(self):
        # 建立動畫，設定等待100個sample的時間
        animation = FuncAnimation(self.fig, self.animate, frames=len(self.raw_data) + self.component_len - self.window_size, interval=75, repeat=True)

        # 經過測試，若顯示動畫後才儲存，會導致程式直接關閉，沒辦法連續執行
        # 所以若無儲存動畫需求，可以直接註解以提升效能
        
        # 儲存動畫
        output_dir = 'output/animations/slidingwindow'
        os.makedirs(output_dir, exist_ok=True)
        raw_data_filename = self.raw_data_path.split('/')[-1].split('.')[0]
        animation.save(os.path.join(output_dir, f'animation_slidingwindow_{raw_data_filename}.gif'), writer='pillow', fps=15)
        
        # 顯示動畫
        # plt.rcParams['animation.html'] = 'jshtml'
        plt.show()

def animation_slidingwindow(index, datasets_dir, window_size=50):
    """
    Generate an animation of the sliding window for the given index.

    Parameters:
    - index (int): The index of the file to be animated.
    - datasets_dir (str): The directory of the datasets.
    - window_size (int): The size of the sliding window. Default is 50.

    Returns:
    None
    """
    ani = Ani_SlidingWindow(index, datasets_dir, window_size=window_size)
    if ani.generate_data():
        ani.generate_static_plot()
        ani.start_animation()