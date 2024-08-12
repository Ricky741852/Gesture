import os
import torch
import numpy as np

from alive_progress import alive_bar
from os import listdir

from src.utils import Color, GroundTruth

os.environ['KMP_DUPLICATE_LIB_OK']='True'

TRAIN_DATA_DIR = 'data/datasets/trainData'  # directory of raw training data
TEST_DATA_DIR = 'data/datasets/testData'  # directory of raw testing data
PROCESSSED_DATASETS_DIR = 'data/processed/datasets'  # directory of formatted data

class GestureData():
    def __init__(self, train_data_path, test_data_path, save_path, window_size, gesture_label=None):
        """
        Initialize the PreProcess object.

        Args:
            train_data_path (str): The path to the training data.
            test_data_path (str): The path to the test data.
            save_path (str): The path to save the pre-processed data.
            window_size (int): The size of the windows for data processing.
            gesture_label (list, optional): The gesture labels. Defaults to [-1000, -1000, -1000, -1000, -1000].
        """
        if gesture_label is None:
            gesture_label = [-1000, -1000, -1000, -1000, -1000]
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        # self.other_data_path = other_data_path
        self.save_path = save_path
        self.accomplish_path = None
        self.data_classes = list()
        self.data_classes_total = 0
        self.window_size = window_size
        self.gesture_label = gesture_label

        self.gesture_raw_data = list()

        self.x = None
        self.y = None

    def _get_file(self, data_path):
        """
        Retrieves the file paths and corresponding labels from the specified data path.

        Args:
            data_path (str): The path to the directory containing the data.

        Returns:
            None
        """
        print(Color.OK + f'Start generating data...' + Color.RESET)
        data_classes = set()
        accomplish_path = []
        for subdir in sorted(listdir(data_path)):   # 檔案排序
            data_path_iter = os.path.join(data_path, subdir) # 資料夾排序
            for i in sorted(listdir(data_path_iter)):
                acc_path = os.path.join(data_path_iter, i)
                accomplish_path.append((acc_path, subdir))
                data_classes.add(subdir)
                # Formatted as ('train1&2_test2/testData2/0/02_2019-01-11-07-51-08_C.Y_Chen.txt', '0')
        self.accomplish_path = accomplish_path
        self.data_classes = sorted(list(data_classes))

        # Classification
        self.data_classes_total = len(self.data_classes)
        # print(f'Data class:  {self.data_classes}')

    def _get_raw_data_from_file(self, path):
        data = list()
        # print(path)
        with open(path, "r") as f:
            raw = f.readline()
            while raw:
                data.append(list(map(int, raw.split(","))))
                raw = f.readline()
        # print(data)
        return data

    def _find_gesture_label(self, data):
        """
        Finds the indices of the gesture labels in the given data and removes them from the data.

        Args:
            data (list): The input data.

        Returns:
            list: The indices of the gesture labels in the data.

        Raises:
            RuntimeError: If the total number of gesture labels is not even.
        """
        N = len(data)
        label = list()
        for i, raw in enumerate(data):
            if raw == self.gesture_label:
                label.append(i)
                del data[i]

        if len(label) % 2 != 0:
            error_message = f"Total gesture label {self.gesture_label} is not even"
            print(Color.FAIL + error_message + Color.RESET)
            raise RuntimeError(error_message)
        else:
            return label

    def _generate_ground_truth(self, data, label):
        """
        Generates the ground truth for the given data and label.

        Args:
            data (list): The input data.
            label (list): The label for the data.

        Returns:
            list: The generated ground truth.

        """
        ground_truth = [0] * len(data)
        if label is None:
            return

        for i in range(0, len(label), 2):
            ground_truth_len = label[i + 1] - label[i]
            gaussian_ground_truth = GroundTruth(ground_truth_len, ground_truth_len / 6)

            for j, truth in enumerate(gaussian_ground_truth.truth):
                ground_truth[j + label[i]] = truth 

        return ground_truth
    
    def generate_test_data(self, index):
        """
        Generates test data for a given index.

        Args:
            index (int): The index of the test data.

        Returns:
            dict: A dictionary containing the generated test data, including the input data, gesture class, and file path.
        """
        if not self.accomplish_path:
            self._get_file(self.test_data_path)
        raw_data = self._get_raw_data_from_file(self.accomplish_path[index][0])

        data_path = self.accomplish_path[index][0]
        print('data_path:', data_path)
        gesture_class = self.accomplish_path[index][1]

        data_len = len(raw_data)
        if data_len < self.window_size:
            print(f"file data {self.accomplish_path[index]} total line smaller than {self.window_size}")

        middle = self.window_size

        if gesture_class != '0':
            # 非背景手勢才有ground_truth
            label = self._find_gesture_label(raw_data)
            ground_truth = self._generate_ground_truth(raw_data, label)

            # 找出手勢中心點
            middle = ground_truth.index(1)

        np_raw_data = np.array(raw_data)

        if middle < self.window_size:
            np_raw_data = np.insert(np_raw_data, 0, np.zeros((self.window_size - middle, len(raw_data[0]))), axis=0)
            middle = self.window_size

        destinition_window_data = np_raw_data[middle - self.window_size:middle]
            
        x = destinition_window_data / 360
        ret = {
            "x": x,
            "accomplish_path": data_path,
            "gesture_class": gesture_class
        }

        return ret

    def generate_data(self, tag):
        """
        Generate formatted training or testing data.

        Args:
            tag (str): The tag indicating whether to generate 'train' or 'test' data.

        Returns:
            None
        """
        # get file path
        if tag == 'train':
            self._get_file(self.train_data_path)
        elif tag == 'test':
            self._get_file(self.test_data_path)
        x = list()
        x_label = list()
        x_path = list()
        y = list()
        data_classes = self.data_classes    # ['0', '1', '2', '3']
        data_classes_total = self.data_classes_total

        test_count = 0

        # for all file
        with alive_bar(len(self.accomplish_path), title=f'Formatting data') as bar:
            for j, path in enumerate(self.accomplish_path):

                # get file data
                raw_data = self._get_raw_data_from_file(path[0])    # path: 'train1&2_test2/testData2/0/02_2019-01-11-07-51-08_C.Y_Chen.txt'
                raw_data_class = path[1] # '0', '1', '2', '3'

                # multi class
                data_classes_index = data_classes.index(raw_data_class) # 取得raw_data_class在data_classes的index 0, 1, 2, 3

                label = self._find_gesture_label(raw_data)
                
                data_len = len(raw_data) # 經過已經移除gesture_label後的資料長度

                # 若資料長度小於windows_size，則跳過此筆資料
                if data_len < self.window_size:
                    print(Color.WARN + f"file data {path} total line smaller than {self.window_size}" + Color.RESET)
                    continue

                # 若是背景種類，因為沒有起始點和終點，所以ground_truth直接設為1
                ground_truth = [1] * data_len

                # 若非背景種類，則依照Label位置調整長度，並產生ground_truth
                if data_classes_index != 0:
                    if label[0] < self.window_size:
                        raw_data = np.insert(raw_data, 0, np.zeros((self.window_size - label[0], len(raw_data[0])), dtype=int), axis=0)
                        label[1] = label[1] + self.window_size - label[0]
                        label[0] = self.window_size

                    ground_truth = self._generate_ground_truth(raw_data, label)
                    data_len = len(raw_data) # 經過已經調整後的資料長度
                
                # split data by slide windows
                for i in range(data_len - self.window_size + 1):
                    window_data = raw_data[i:i + self.window_size] # i = 0: => 0~49, 1: => 1~50, 2: => 2~51, ...
                    window_ground_truth = ground_truth[i + self.window_size - 1]   # i = 0: => 49, 1: => 50, 2: => 51, ...
                    x.append(np.array(window_data) / 360)
                    x_label.append(raw_data_class)
                    x_path.append(path)

                    all_classes_y = np.zeros((data_classes_total), dtype=np.float32)
                    all_classes_y[data_classes_index] = window_ground_truth

                    # 背景手勢的分數為 1 - 真實手勢的分數
                    if data_classes_index != 0:
                        all_classes_y[0] = 1 - window_ground_truth

                    test_count += 1
                    y.append(all_classes_y.T)
                bar()

        # Format raw data to training data
        self.x = {"x": np.array(x), "path": x_path, "label": x_label}
        self.y = {"y": np.array(y)}
        print(Color.MSG + f'gesture_data.x: {self.x.keys()}' + Color.RESET)  # dict_keys(['x', 'path', 'label'])
        print(Color.MSG + f'gesture_data.y: {self.y.keys()}' + Color.RESET)  # dict_keys(['y'])
        print(Color.OK + f'Successfully formatted original raw {tag} data!' + Color.RESET)

        data_dict = {'dataX': self.x, 'dataY': self.y}
        save_pt_path = os.path.join(self.save_path, f'format_{tag}_data.pt')
        torch.save(data_dict, save_pt_path)
        print(Color.H_OK + f'Successfully saved {tag} data to {save_pt_path}!' + Color.RESET)

def data_preprocess(window_size):
    """Pre-process and save all data to format_data/ as pt file"""
    gesture_data = GestureData(TRAIN_DATA_DIR, TEST_DATA_DIR, PROCESSSED_DATASETS_DIR, window_size=window_size)

    # Generate training data, testing data
    gesture_data.generate_data(tag='train')
    gesture_data.generate_data(tag='test')

def test_preprocess(window_size, index):
    """For getting full test gesture data, each piece of data is made from a complete txt file"""

    # Get full test data
    gesture_data = GestureData(None, TEST_DATA_DIR, PROCESSSED_DATASETS_DIR, window_size=window_size)
    test_data = gesture_data.generate_test_data(index=index)
    return test_data