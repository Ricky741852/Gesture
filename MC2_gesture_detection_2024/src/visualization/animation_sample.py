import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation
from collections import deque

os.environ['KMP_DUPLICATE_LIB_OK']='True'

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

from src.data import GestureDataHandler
from src.models import GestureDetector

BACK_GROUND_CLASS = 0

class Ani_Sample():
    def __init__(self, index, model_name, datasets_dir, window_size=50):
        # 初始化手勢辨識模型
        self.Gesture_model = GestureDetector(model_name, window_size=window_size)

        # 取得測試資料
        data_handler = GestureDataHandler(datasets_dir, window_size=window_size)

        self.window_size = window_size
        self.raw_data, self.gesture_label, self.ground_truth, self.gesture_class, self.raw_data_path = data_handler.generate_test_data(index)

        # Raw data plottings
        self.raw_class_list = ['1', '2', '3', '4', '5']
        self.raw_colors = ["orange", "blue", "red", "brown", "green"]

        # Prediction data plottings
        self.gesture_class_list = ['0', '1', '2', '3']
        self.gesture_colors = ["purple", "orange", "blue", "red"]
        
        # 每個分類的真實分數
        self.ground_truths = [[0 for _ in range(len(self.raw_data) + (self.window_size) - 1)] for _ in range(len(self.gesture_class_list))]
        
        # 取得每個分類的預測分數
        self.predict_data = [[list() for _ in range(self.window_size)] for _ in range(len(self.gesture_class_list))]
        
        # 用於判斷手勢類別
        self.is_in_Gesture = False   # 是否正在進行手勢
        self.current_class = 0
        self.last_scores = deque([0, 0, 0], maxlen=3)

        # 使用 GridSpec 來手動調整子圖布局
        self.fig = plt.figure(figsize=(15, 15))
        gs = GridSpec(3, 1, height_ratios=[1, 1, 1])

        self.ax1 = self.fig.add_subplot(gs[0, 0])
        self.ax2 = self.fig.add_subplot(gs[1, 0])
        self.ax3 = self.fig.add_subplot(gs[2, 0])

        # 為每個子圖設置 y 軸標籤
        self.ax1.set_ylabel('Raw Data', fontsize=25)
        self.ax2.set_ylabel('Ground Truth', fontsize=25)
        self.ax3.set_ylabel('Predicted Data', fontsize=25)

        # 畫出原始數據的線條
        self.lines_raw = [self.ax1.plot([], [], lw=2, label=f'Sensor {raw_num}', color=color)[0] 
                          for raw_num, color in zip(self.raw_class_list, self.raw_colors)]
        self.ax1.set_xlim(0, self.window_size)
        self.ax1.set_ylim(-110, 110)  # 假設 y 軸範圍在 -200 到 400 之間
        self.ax1.legend(loc='lower left', bbox_to_anchor=(1, 0.4), fontsize=20)

        # 畫出Ground Truth數據的線條
        self.lines_truth = [self.ax2.plot([], [], lw=2, label=f'Gesture {class_num}', color=color)[0]
                              for class_num, color in zip(self.gesture_class_list, self.gesture_colors)]
        self.ax2.set_xlim(0, self.window_size)
        self.ax2.set_ylim(-0.05, 1.05)
        self.ax2.legend(loc='lower left', bbox_to_anchor=(1, 0.5), fontsize=20)

        # 畫出預測數據的線條
        self.lines_predict = [self.ax3.plot([], [], lw=2, label=f'Gesture {class_num}', color=color)[0]
                              for class_num, color in zip(self.gesture_class_list, self.gesture_colors)]
        self.ax3.set_xlim(0, self.window_size)
        self.ax3.set_ylim(-0.05, 1.05)  # 假設 y 軸範圍在 0 到 1 之間
        self.ax3.legend(loc='lower left', bbox_to_anchor=(1, 0.5), fontsize=20)

        self.predict_class = self.ax3.text(.5, .5, '', fontsize=15)

        # 初始空的視窗
        x = np.arange(0, self.window_size)
        y = np.zeros(self.window_size)
        
        for line in self.lines_raw:
            line.set_data(x, y)

        for line in self.lines_truth:
            line.set_data(x, y)

        for line in self.lines_predict:
            line.set_data(x, y)

        # 調整子圖布局
        plt.tight_layout(rect=[0, 0, 1, 1])
        
    def generate_data(self):
            """
            Generates data for gesture recognition.

            Returns:
                bool: True if data generation is successful, False otherwise.
            """
            if len(self.raw_data) < self.window_size:
                print("Data less than {}".format(self.window_size))
                return False
            
            # Raw data
            self.np_data = np.array(self.raw_data)
            self.np_data = np.insert(self.np_data, 0, np.zeros((self.window_size, len(self.raw_class_list))), axis=0)

            # Ground Truth data
            self.ground_truths[self.gesture_class_list.index(self.gesture_class)] = self.ground_truth
            self.ground_truths[0] = [1 - gt for gt in self.ground_truth]

            for i in range(1, len(self.gesture_class_list)):
                self.ground_truths[i] = [0] * (self.window_size - 1) + self.ground_truths[i]
            self.ground_truths[0] = [1] * (self.window_size - 1) + self.ground_truths[0]

            # Prediction data
            data_with_zeros_front = np.insert(self.np_data, 0, np.zeros((self.window_size -1, len(self.raw_class_list))), axis=0)
            data = data_with_zeros_front/360
            self.windows = self.Gesture_model.make_sliding_windows(data)
            self.predict_data = self.Gesture_model.predict(self.windows)

            return True

    def animate(self, frame):
        """
        Animates the gesture recognition visualization.

        Args:
            frame (int): The current frame number.

        Returns:
            tuple: A tuple containing the updated line objects for the raw data, ground truth data, and predicted data.
        """
        window_size = self.window_size
        current_class = self.current_class

        # 處理原始數據
        raw_data_T = self.np_data[frame:frame+window_size].T
        for i in range(len(self.raw_class_list)):
            self.lines_raw[i].set_ydata(raw_data_T[i])

        # 處理 Ground Truth 數據            
        ground_truth = self.ground_truths
        for i in range(len(self.gesture_class_list)):
            self.lines_truth[i].set_ydata(ground_truth[i][frame:frame+window_size])

        # # 處理預測數據
        predict_data_T = np.array(self.predict_data[frame:frame+window_size]).T
        for i in range(len(self.gesture_class_list)):
            self.lines_predict[i].set_ydata(predict_data_T[i])

        # 獲取當前的預測標籤
        current_predict_data = self.predict_data[frame + window_size]
        argmax_score = np.argmax(current_predict_data)
        max_score = np.max(current_predict_data)

        if not self.is_in_Gesture:
            if argmax_score != 0 and max_score >= 0.7:
                self.last_scores.append(argmax_score)
                if self.last_scores[0] == self.last_scores[1] == self.last_scores[2]:
                    self.is_in_Gesture = True
                    self.current_class = self.last_scores[0]
            else:
                self._reset_last_scores()
        else:
            if current_predict_data[current_class] < 0.3:
                self.is_in_Gesture = False
                self._reset_last_scores()

        # 更新預測手勢的文本
        self.predict_class.set_text(f"predict gesture : {current_class == BACK_GROUND_CLASS and 'None' or (current_class)}")

        return self.lines_raw, self.lines_truth, self.lines_predict
    
    def _reset_last_scores(self):
        self.current_class = BACK_GROUND_CLASS
        self.last_scores = deque([BACK_GROUND_CLASS, BACK_GROUND_CLASS, BACK_GROUND_CLASS], maxlen=3)

    def start_animation(self):
        # 建立動畫，設定等待100個sample的時間
        animation = FuncAnimation(self.fig, self.animate, frames=len(self.raw_data), interval=50, repeat=True)

        # 顯示動畫
        plt.rcParams['animation.html'] = 'jshtml'
        plt.show()
        
        # 儲存動畫
        # output_dir = 'output/animations/sample'
        # os.makedirs(output_dir, exist_ok=True)
        # raw_data_filename = self.raw_data_path.split('/')[-1].split('.')[0]
        # animation.save(os.path.join(output_dir, f'animation_sample_{raw_data_filename}.gif'), writer='pillow', fps=20)

def animation_sample(index, model_name, datasets_dir, window_size=50):
    """
    Plot the animation based on the given index, model name, datasets directory, and window size.

    Parameters:
    - index (int): The index of the data to be plotted.
    - model_name (str): The name of the model to be used for prediction.
    - datasets_dir (str): The directory where the datasets are stored.
    - window_size (int): The size of the sliding window. Default is 50.

    Returns:
    None
    """
    ani = Ani_Sample(index, model_name, datasets_dir, window_size=window_size)
    if ani.generate_data():
        ani.start_animation()