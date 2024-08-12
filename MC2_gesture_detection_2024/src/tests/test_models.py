import time
import os
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
from collections import deque
import numpy as np
import matplotlib.pyplot as plt

from src.data import test_preprocess, GestureDataHandler
from src.models import GestureDetector
from src.utils import Color, check_directories

plt.rcParams["font.family"] = "Times New Roman"
timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))

class TestModels:
    def __init__(self, model_evaluation, datasets_dir, gesture_class_num, window_size, index=0, all_data_count=400) -> None:
        """
        Initializes the GestureTestModels class.
        Parameters:
        - model_evaluation (str): The evaluation model to be used for gesture recognition.
        - datasets_dir (str): The directory path of the datasets.
        - gesture_class_num (int): The number of gesture classes.
        - window_size (int): The size of the sliding window for data processing.
        - index (int, optional): The index of the data. Defaults to 0.
        - all_data_count (int, optional): The total count of data. Defaults to 400.
        """
        # 初始化手勢辨識模型
        self.Gesture_model = GestureDetector(model_evaluation, window_size=window_size)

        # 取得測試資料
        self.data_handler = GestureDataHandler(datasets_dir, window_size=window_size)

        self.model_evaluation = model_evaluation
        self.gesture_class_num = gesture_class_num
        self.window_size = window_size
        self.index = index
        self.all_data_count = all_data_count
        self.answer_label_list, self.output_label_list = [], []

    def test_all(self):
        """
        Test all the gesture files in the dataset.
        Returns:
            None
        """
        self.answer_label_list, self.output_label_list = [], []
        correct = 0

        file_start, file_end = 0, self.all_data_count

        for idx in range(file_start, file_end):
            """Each piece of txt test data belongs to an individual index"""
            if(self.test_single(idx)):
                correct += 1

        # Calculate average data length of 72 test gesture files
        print(f'Window size: {self.window_size} (Sampling rate: 50/s)')

        accuracy = correct / self.all_data_count
        print(
            Color.MSG
            + f'Accuracy of test data: {accuracy}'
            + Color.RESET
        )
        
    def test_single(self, index):
        """
        Test a single gesture detection.
        Args:
            index (int): The index of the test data.
        Returns:
            bool: True if the gesture is predicted correctly, False otherwise.
        """
        raw_data, gesture_label, ground_truth, answer, raw_data_path = self.data_handler.generate_test_data(index)

        if len(raw_data) < self.window_size:
            print("Data less than {}".format(self.window_size))
            return False
        
        predict_gesture = self.detection_gesture(raw_data)

        print(Color.H_OK + f'Gesture answer: {answer}', end=" " + Color.RESET)
        print(Color.H_MSG + f'Gesture predict: {predict_gesture}' + Color.RESET)
        self.answer_label_list.append(answer)
        self.output_label_list.append(predict_gesture)

        if int(answer) == predict_gesture:
            print(Color.H_OK + f'Predict the gesture correctly' + Color.RESET)
            return True
        else:
            print(Color.H_FAIL + f'Predict the gesture incorrectly' + Color.RESET)
            return
        
    def detection_gesture(self, raw_data) -> int:
        """
        Detects the gesture from the given raw data.
        Args:
            raw_data (list): The raw data to be processed.
        Returns:
            int: The detected gesture class. Returns 0 if no gesture is detected.
        """
        # code implementation...
        raw_data_len = len(raw_data)

        raw_class_num = 5

        # Raw data
        np_data = np.array(raw_data)
        
        # Prediction data
        data_with_zeros_front = np.insert(np_data, 0, np.zeros((self.window_size -1, raw_class_num)), axis=0)
        data = data_with_zeros_front/360
        windows = self.Gesture_model.make_sliding_windows(data)
        predict_data = self.Gesture_model.predict(windows)

        # 用於判斷手勢類別
        is_in_Gesture = False   # 是否正在進行手勢
        current_class = 0
        last_scores = deque([0, 0, 0], maxlen=3)

        detection = []

        for i in range(raw_data_len):
            current_predict_data = predict_data[i]
            argmax_score = np.argmax(current_predict_data)
            max_score = np.max(current_predict_data)

            if not is_in_Gesture:
                if argmax_score != 0 and max_score >= 0.7:
                    last_scores.append(argmax_score)
                    if last_scores[0] == last_scores[1] == last_scores[2]:
                        is_in_Gesture = True
                        current_class = last_scores[0]
                else:
                    current_class = 0
                    last_scores = deque([0, 0, 0], maxlen=3)
            else:
                if current_predict_data[current_class] < 0.3:
                    detection.append(current_class)
                    is_in_Gesture = False
                    current_class = 0
                    last_scores = deque([0, 0, 0], maxlen=3)
        
        return 0 if len(detection) == 0 else detection[0]
    
    def plot_confusion_matrix(self):
        """
        Plot the confusion matrix for the classification results.
        This function generates a confusion matrix based on the true and predicted labels,
        and plots it using matplotlib. It also saves a classification report to a text file.
        Parameters:
        - None
        Returns:
        - None
        """
        cm_dir = 'output/confusion_matrices'
        check_directories(cm_dir)

        y_true = list(map(int, self.answer_label_list))
        y_pred = self.output_label_list

        # Generate classification report of precision and recall
        class_list = [
            "Gesture " + str(i) for i in range(self.gesture_class_num)
        ]  # Gesture 1, ..., Gesture n
        print(Color.H_INFO + f'Classification Report ' + '=' * 60 + Color.RESET)
        print(classification_report(y_true, y_pred, target_names=class_list))

        # Save report to confusion_matrix/20230426_classification_report.txt
        report = os.path.join(cm_dir, f'{timestamp}_classification_report.txt')
        with open(report, 'w') as f:
            print(
                classification_report(y_true, y_pred, target_names=class_list), file=f
            )

        # Plot confusion matrix and save, choose seaborn or matplotlib
        confusion_mat = confusion_matrix(y_true, y_pred)

        # Plot confusion matrix with values
        plt.imshow(confusion_mat, cmap='Blues')

        # Add text annotations to each cell
        thresh = confusion_mat.max() / 2.0
        for i in range(confusion_mat.shape[0]):
            for j in range(confusion_mat.shape[1]):
                plt.text(j, i, format(confusion_mat[i, j], 'd'),
                         horizontalalignment="center",
                         color="white" if confusion_mat[i, j] > thresh else "black",
                         fontsize=10)

        plt.xticks(range(self.gesture_class_num), class_list, fontsize=10)
        plt.yticks(range(self.gesture_class_num), class_list, fontsize=12)
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold', )
        plt.xlabel('Predicted Gesture', fontsize=14, fontweight='bold')
        plt.ylabel('True Gesture', fontsize=12, fontweight='bold')
        savefig_name = f'{timestamp}_confusion_matrix.png'
        plt.savefig(os.path.join(cm_dir, savefig_name))
        plt.show()

    def test_single_with_highest_score_window(self, index):
        """
        NOT USED
        Test the model's prediction for a single window with the highest score.
        Parameters:
        - index (int): The index of the testing data.
        Returns:
        - bool: True if the model predicts the gesture correctly, False otherwise.
        """
        # Load testing data
        test_data = test_preprocess(window_size=self.window_size, index=index)
        data, answer = test_data['x'], test_data['gesture_class']

        # 因為模型的輸入是三維的(None, 50, 5)，所以需要對輸入的數據進行reshape
        reshaped_data = data.reshape(1, data.shape[0], data.shape[1])

        prediction = self.Gesture_model.predict(reshaped_data, verbose=0)
        predict_gesture = np.argmax(prediction)

        print(Color.H_OK + f'Gesture answer: {answer}', end=" " + Color.RESET)
        print(Color.H_MSG + f'Gesture predict: {predict_gesture}' + Color.RESET)
        self.answer_label_list.append(answer)
        self.output_label_list.append(predict_gesture)

        if int(answer) == predict_gesture:
            print(Color.H_OK + f'Predict the gesture correctly' + Color.RESET)
            return True
        else:
            print(Color.H_FAIL + f'Predict the gesture incorrectly' + Color.RESET)
            return False