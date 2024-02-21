# import tensorflow as tf
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#     except RuntimeError as e:
#         print(e)

from tensorflow.keras.models import load_model
import numpy as np

# from loss import focal_loss,Index_L1_loss




class Gesture_Detection():
    def __init__(self,model_name,windows_size=100):
        """
        Initialize the Gesture_Detection class.

        Parameters:
        - model_name (str): The name of the model to be loaded.
        - windows_size (int): The size of the sliding windows.

        Returns:
        None
        """
        self.windows_size = windows_size
        self.model = load_model(f"./saved_model_h5/{model_name}.h5")

    def predict(self,windows):
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
        for i in range(0, len(data) - self.windows_size + 1, stride):
            window = data[i: i + self.windows_size]
            windows.append(window)
        windows = np.array(windows)
        print(f'Window shape: {windows.shape}')
        return windows
    
    


