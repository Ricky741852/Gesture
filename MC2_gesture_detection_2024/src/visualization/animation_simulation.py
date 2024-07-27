import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import Levenshtein
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
from src.models.gesture_detector import GestureDetector

class Ani_Simulation():
    def __init__(self, index, model_name, Gesture_Data_Model, input_string, windows_size=50):
        self.window_size = windows_size
        self.raw_data, self.raw_data_path = Gesture_Data_Model.generate_simulate_data(index)

        # Raw data plottings
        self.raw_class_list = ['1', '2', '3', '4', '5']
        self.raw_colors = ["orange", "blue", "red", "brown", "green"]

        # Prediction data plottings
        self.gesture_class_list = ['0', '1', '2', '3']
        self.gesture_colors = ["purple", "orange", "green", "red"]
        
        # 取得每個分類的最高分數
        self.predict_data = [[list() for _ in range(self.window_size)] for _ in range(len(self.gesture_class_list))]
        
        # 用於判斷手勢類別
        self.is_in_Gesture = False   # 是否正在進行手勢
        self.current_class = 3
        self.last_scores = deque([3, 3, 3], maxlen=3)

        self.input_string = input_string
        self.predict_string = str()

        self.Gesture_model = GestureDetector(model_name, window_size=windows_size)

        # 初始化畫布和軸
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(15, 10))

        # 為每個子圖設置 y 軸標籤
        self.ax1.set_ylabel('Raw Data', fontsize=14)
        self.ax2.set_ylabel('Predicted Data', fontsize=14)

        # 畫出原始數據的線條
        self.lines_raw = [self.ax1.plot([], [], lw=2, label=f'Sensor {raw_num}', color=color)[0] 
                          for raw_num, color in zip(self.raw_class_list, self.raw_colors)]
        self.ax1.set_xlim(0, self.window_size)
        self.ax1.set_ylim(-110, 110)
        self.ax1.legend(bbox_to_anchor=(0.8, 0.8, 0.3, 0.2), loc='upper right')

        # 畫出預測數據的線條
        self.lines_predict = [self.ax2.plot([], [], lw=2, label=f'Gesture {class_num}', color=color)[0]
                              for class_num, color in zip(self.gesture_class_list, self.gesture_colors)]
        self.ax2.set_xlim(0, self.window_size)
        self.ax2.set_ylim(-0.05, 1.05)
        self.ax2.legend(bbox_to_anchor=(0.8, 0.8, 0.3, 0.2), loc='upper right')

        self.predict_class = self.ax2.text(.5, .5, '', fontsize=15)

        # 初始空的視窗
        x = np.arange(0, self.window_size)
        y = np.zeros(self.window_size)
        
        for line in self.lines_raw:
            line.set_data(x, y)

        for line in self.lines_predict:
            line.set_data(x, y)
        
    def generate_data(self):

        if len(self.raw_data) < self.window_size:
            print("Data less than {}".format(self.window_size))
            return False
        
        self.np_data = np.array(self.raw_data)
        self.np_data = np.insert(self.np_data, 0, np.zeros((self.window_size, len(self.raw_class_list))), axis=0)

        # 最前方補上49個0，以產生第一筆raw data的window，進而預測第一個分數
        data_with_zeros_front = np.insert(self.np_data, 0, np.zeros((self.window_size - 1, len(self.raw_class_list))), axis=0)

        data = data_with_zeros_front/360

        self.windows = self.Gesture_model.make_sliding_windows(data)
        self.predict_data = self.Gesture_model.predict(self.windows)

        return True

    def animate(self, frame):
        window_size = self.window_size
        current_class = self.current_class
        if frame <= len(self.raw_data):
            # 處理原始數據
            raw_data_T = self.np_data[frame:frame+window_size].T
            for i in range(len(self.raw_class_list)):
                self.lines_raw[i].set_ydata(raw_data_T[i])

            # 處理預測數據
            predict_data_T = np.array(self.predict_data[frame:frame+window_size]).T
            for i in range(len(self.gesture_class_list)):
                self.lines_predict[i].set_ydata(predict_data_T[i])
                
            # 獲取當前的預測標籤
            current_predict_data = self.predict_data[frame + window_size - 1][1:]
            max_score = np.max(current_predict_data)
            argmax_score = np.argmax(current_predict_data)

            back_ground_class = 3
            
            if not self.is_in_Gesture:
                if max_score > 0.6:
                    self.last_scores.append(argmax_score)
                    if self.last_scores[0] == self.last_scores[1] == self.last_scores[2] != back_ground_class:
                        self.is_in_Gesture = True
                        self.current_class = self.last_scores[0]
                else:
                    self.last_scores.append(back_ground_class)
            else:
                if current_predict_data[current_class] < 0.3:
                    self.is_in_Gesture = False
                    self.predict_string += str(current_class + 1)
                    self.current_class = back_ground_class
                    self.last_scores = deque([back_ground_class, back_ground_class, back_ground_class], maxlen=3)

            # 更新預測手勢的文本
            self.predict_class.set_text(f"predict gesture : {current_class == back_ground_class and 'None' or (current_class + 1)}")

        return self.lines_raw, self.lines_predict

    def start_animation(self):
        # 建立動畫，設定等待100個sample的時間
        animation = FuncAnimation(self.fig, self.animate, frames=(len(self.raw_data) + self.window_size), interval=50, repeat=False)

        # 儲存動畫
        output_dir = 'output/animations/simulation'
        os.makedirs(output_dir, exist_ok=True)
        raw_data_filename = self.raw_data_path.split('/')[-1].split('.')[0]
        animation.save(os.path.join(output_dir, f'animation_simulation_{raw_data_filename}.gif'), writer='pillow', fps=20)
        
        # 因為儲存時就會先計算一次，所以在顯示之前先計算
        print(self.editdistance())

        # 顯示動畫
        plt.rcParams['animation.html'] = 'jshtml'
        plt.show()

    def editdistance(self):
        print(self.input_string, self.predict_string)
        return Levenshtein.ratio(self.input_string, self.predict_string)
