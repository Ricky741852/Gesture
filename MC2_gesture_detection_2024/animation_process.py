from os import listdir
from utility import GroundTruth

class Gesture_Data():
    def __init__(self,path,windows_size=128,gesture_label=[-1000,-1000,-1000,-1000,-1000]):
        self.path = path
        self.accomplish_path = None # file path of every data
        self.gesture_class_list = list()  # classification list
        self.windows_size = windows_size
        self.gesture_label = gesture_label
        self.gesture_raw_data = list()
    
    def _get_file(self):
        """
        Retrieves the file paths and gesture class labels from the specified directory.

        Returns:
            None
        """
        gesture_class_list = set()
        accomplish_path = []
        path = self.path
        for subdir in sorted(listdir(path)):
            for i in sorted(listdir(path + '/' + subdir)):
                accomplish_path.append((path + '/' + subdir + '/' + i, subdir))
                gesture_class_list.add(subdir)
                # 格式如：('train1&2_test2/testData2/0/02_2019-01-11-07-51-08_C.Y_Chen.txt', '0')
        self.accomplish_path = accomplish_path        
        self.gesture_class_list = sorted(list(gesture_class_list))

    def _get_raw_data_from_file(self, path):
        """
        Read raw data from a file and return it as a list of lists.

        Args:
            path (str): The path to the file containing the raw data.

        Returns:
            list: A list of lists, where each inner list represents a line of raw data.

        """
        raw_data = list()
        with open(path, "r") as f:
            raw = f.readline()
            while raw:
                raw_data.append(list(map(int, raw.split(","))))
                raw = f.readline()
        return raw_data

    def _find_gesture_label(self, data):
        """
        Find the indices of the gesture labels in the given data.

        Args:
            data (list): The data to search for gesture labels.

        Returns:
            list: A list of indices where the gesture labels are found.

        Raises:
            RuntimeError: If the total number of gesture labels is not even.

        """
        label = list()
        for i, raw in enumerate(data):
            if raw == self.gesture_label:
                label.append(i)
                del data[i]

        if len(label) % 2 != 0:
            print(f"Total gesture label {self.gesture_label} is not even")
            raise RuntimeError(f"Total gesture label {self.gesture_label} is not even")
        else:
            return label

    def _generate_ground_truth(self, data, label):
        """
        Generate the ground truth for the given data and label.

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
        Generate test data for a given index.

        Args:
            index (int): The index of the test data.

        Returns:
            tuple: A tuple containing the following elements:
                - raw_data (list): The raw data obtained from the file.
                - label (str): The gesture label.
                - grund_truth (list): The ground truth data generated from the raw data.
                - gesture_class_list (list): The list of available data classes.
                - gesture_class (str): The name of the class for the given index.

        Raises:
            None

        """
        if not self.accomplish_path:
            self._get_file()
        raw_data = self._get_raw_data_from_file(self.accomplish_path[index][0])
        data_len = len(raw_data)
        if data_len < self.windows_size:
            print(f"file data {self.accomplish_path[index]} total line smaller than {self.windows_size}")

        gesture_label = self._find_gesture_label(raw_data)
        print(gesture_label)
        grund_truth = self._generate_ground_truth(raw_data, gesture_label)

        gesture_class_list = self.gesture_class_list
        gesture_class = self.accomplish_path[index][1]

        return raw_data, gesture_label, grund_truth, gesture_class_list, gesture_class