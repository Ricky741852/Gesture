from tensorflow.keras.models import load_model
import os
import numpy as np

class GestureDetector():
    """
    載入預測模型
    """
    def __init__(self,model_name,window_size=50):
        """
        Initialize the Gesture_Detection class.

        Parameters:
        - model_name (str): The name of the model to be loaded.
        - windows_size (int): The size of the sliding windows.

        Returns:
        None
        """
        self.window_size = window_size
        model_path = os.path.join(os.path.dirname(__file__), '..', '..', 'output', 'models', f"{model_name}.h5")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file does not exist at: {model_path}")
        self.model = load_model(model_path)

    def predict(self, windows):
        """
        Perform gesture prediction on the given windows.

        Parameters:
        - windows (numpy.ndarray): The sliding windows of data.

        Returns:
        numpy.ndarray: The predictions for each window.
        """
        predictions = self.model.predict(windows, verbose=0)
        return predictions

    def make_sliding_windows(self, data):
        """
        Generate sliding windows from the given data.

        Parameters:
        - data (numpy.ndarray): The input data.

        Returns:
        numpy.ndarray: The sliding windows.

        Example:
        Sampling rate = 50Hz, windwows_stride means the step size of the window sliding.
        For example: windows_size = 100, windows_stride = 50, it means that the window slides per second.
        """
        stride = 1

        # Generate sliding windows
        windows = []
        for i in range(0, len(data) - self.window_size + 1, stride):
            window = data[i: i + self.window_size]
            windows.append(window)
        windows = np.array(windows)
        # print(f'Window shape: {windows.shape}')
        return windows
    
    


