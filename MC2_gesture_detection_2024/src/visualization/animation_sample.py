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
from collections import deque
from ..models.gesture_detector import GestureDetector

BACK_GROUND_CLASS = 0

class Ani_Sample():
    def __init__(self, index, model_name, Gesture_Data_Model, windows_size=50):
        self.window_size = windows_size
        self.raw_data, self.gesture_label, self.ground_truth, self.gesture_class, self.raw_data_path = Gesture_Data_Model.generate_test_data(index)

        # Raw data plottings
        self.raw_class_list = ['1', '2', '3', '4', '5']
        self.raw_colors = ["orange", "blue", "red", "brown", "green"]

        # Prediction data plottings
        self.gesture_class_list = ['0', '1', '2', '3']
        self.gesture_colors = ["purple", "orange", "blue", "red"]
        
        # 取得每個分類的最高分數
        self.predict_data = [[list() for _ in range(self.window_size)] for _ in range(len(self.gesture_class_list))]
        
        # 用於判斷手勢類別
        self.is_in_Gesture = False   # 是否正在進行手勢
        self.current_class = BACK_GROUND_CLASS
        self.last_scores = deque([BACK_GROUND_CLASS, BACK_GROUND_CLASS, BACK_GROUND_CLASS], maxlen=3)

        self.Gesture_model = GestureDetector(model_name, window_size=windows_size)

        # 初始化畫布和軸
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(3, 1, figsize=(15, 15))

        # 畫出原始數據的線條
        self.lines_raw = [self.ax1.plot([], [], lw=2, label=f'Sensor {raw_num}', color=color)[0] 
                          for raw_num, color in zip(self.raw_class_list, self.raw_colors)]
        self.ax1.set_xlim(0, self.window_size)
        self.ax1.set_ylim(-110, 110)  # 假設 y 軸範圍在 -200 到 400 之間
        self.ax1.legend(bbox_to_anchor=(0.8, 0.8, 0.3, 0.2), loc='upper right')

        # 畫出Ground Truth數據的線條
        self.lines_truth = self.ax2.plot([], [], lw=2, label=f'Ground Truth')[0]
        self.ax2.set_xlim(0, self.window_size)
        self.ax2.set_ylim(0, 1)
        self.ax2.legend(bbox_to_anchor=(0.8, 0.8, 0.3, 0.2), loc='upper right')

        # 畫出預測數據的線條
        self.lines_predict = [self.ax3.plot([], [], lw=2, label=f'Gesture {class_num}', color=color)[0]
                              for class_num, color in zip(self.gesture_class_list, self.gesture_colors)]
        self.ax3.set_xlim(0, self.window_size)
        self.ax3.set_ylim(0, 1)  # 假設 y 軸範圍在 0 到 1 之間
        self.ax3.legend(bbox_to_anchor=(0.8, 0.8, 0.3, 0.2), loc='upper right')

        self.predict_class = self.ax3.text(.5, .5, '', fontsize=15)

        # 初始空的視窗
        x = np.arange(0, self.window_size)
        y = np.zeros(self.window_size)
        
        for line in self.lines_raw:
            line.set_data(x, y)

        self.lines_truth.set_data(x, y)

        for line in self.lines_predict:
            line.set_data(x, y)
        
    def generate_data(self):
            """
            Generates data for gesture recognition.

            Returns:
                bool: True if data generation is successful, False otherwise.
            """
            if len(self.raw_data) < self.window_size:
                print("Data less than {}".format(self.window_size))
                return False
            
            self.np_data = np.array(self.raw_data)
            # 最前方補上49個0，以產生第一筆raw data的window，進而預測第一個分數
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

        if frame + window_size <= len(self.raw_data):
            # 處理原始數據
            raw_data_T = self.np_data[frame:frame+window_size].T
            for i in range(len(self.raw_class_list)):
                self.lines_raw[i].set_ydata(raw_data_T[i])

            # 處理 Ground Truth 數據
            y_truth = self.ground_truth[frame:frame+window_size]
            self.lines_truth.set_ydata(y_truth)

            # 處理預測數據
            predict_data_T = np.array(self.predict_data[frame:frame+window_size]).T
            for i in range(len(self.gesture_class_list)):
                self.lines_predict[i].set_ydata(predict_data_T[i])

            # 獲取當前的預測標籤
            current_predict_data = self.predict_data[frame + window_size - 1][1:]
            max_score = np.max(current_predict_data)
            argmax_score = np.argmax(current_predict_data)

            if not self.is_in_Gesture:
                if max_score > 0.8:
                    self.last_scores.append(argmax_score)
                else:
                    self.last_scores.append(BACK_GROUND_CLASS)
                if self.last_scores[0] == self.last_scores[1] == self.last_scores[2] != BACK_GROUND_CLASS:
                    self.is_in_Gesture = True
                    self.current_class = self.last_scores[0]
            else:
                if current_predict_data[current_class] < 0.3:
                    self.is_in_Gesture = False
                    self.current_class = BACK_GROUND_CLASS
                    self.last_scores = deque([BACK_GROUND_CLASS, BACK_GROUND_CLASS, BACK_GROUND_CLASS], maxlen=3)

            # 更新預測手勢的文本
            self.predict_class.set_text(f"predict gesture : {current_class == BACK_GROUND_CLASS and 'None' or (current_class + 1)}")

        return self.lines_raw, self.lines_truth, self.lines_predict

    def start_animation(self):
        # 建立動畫，設定等待100個sample的時間
        animation = FuncAnimation(self.fig, self.animate, frames=len(self.raw_data) - self.window_size + 1, interval=50, repeat=True)

        # 顯示動畫
        plt.rcParams['animation.html'] = 'jshtml'
        plt.show()
        
        # 儲存動畫
        output_dir = 'output/animations/sample'
        os.makedirs(output_dir, exist_ok=True)
        raw_data_filename = self.raw_data_path.split('/')[-1].split('.')[0]
        animation.save(os.path.join(output_dir, f'animation_sample_{raw_data_filename}.gif'), writer='pillow', fps=20)