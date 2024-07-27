# __init__.py

# This is the initialization file for the visualization module.

# Import any necessary modules or packages here.

# Define any global variables or constants here.

# Define any functions or classes here.

# Optionally, you can include a main function or code block here.

# This file is executed when the visualization module is imported.

from ..utils.serial_usb import serialUSB
from ..models.gesture_detector import GestureDetector
from ..data.loaders import GestureDataHandler
from .animation_realtime import Ani_Realtime
from .animation_sample import Ani_Sample
from .animation_simulation import Ani_Simulation
from .animation_slidingwindow import Ani_SlidingWindow
from .image_rawdata import Image_RawData
from .image_groundtruth import Image_GroundTruth
from .image_sample import Image_Sample
from .image_slidingwindow import Image_SlidingWindow

class Visualization:
    def __init__(self, model_name, datasets_dir, windows_size=50):
        self.windows_size = windows_size
        self.model_name = model_name
        self.datasets_dir = datasets_dir

    def animation(self, visualization_type="sample"):
        if visualization_type == "realtime":
                self.realtime()
                exit()
        while True:
            index = int(input("-1 to quit : "))

            if index == -1:
                break        
            
            if visualization_type == "sample":
                G = GestureDataHandler(self.datasets_dir, windows_size=self.windows_size) 
                ani = Ani_Sample(index, self.model_name, G, windows_size=self.windows_size)
                if ani.generate_data():
                    ani.start_animation()
            elif visualization_type == "simulation":
                G = GestureDataHandler('data/datasets/simulateData', windows_size=self.windows_size) 
                input_string = input("Enter the GESTURES: ")
                ani = Ani_Simulation(index, self.model_name, G, input_string, windows_size=self.windows_size)
                if ani.generate_data():
                    ani.start_animation()
            elif visualization_type == "slidingwindow":
                G = GestureDataHandler(self.datasets_dir, windows_size=self.windows_size) 
                ani = Ani_SlidingWindow(index, self.model_name, G, windows_size=self.windows_size)
                if ani.generate_data():
                    ani.generate_static_plot()
                    ani.start_animation()
            # if the visualization type is not recognized, print an error message and raise an exception
            else:
                print("Invalid visualization type. Please choose from 'realtime', 'sample', 'rawdata', or 'groundtruth'.")
                raise ValueError("Invalid visualization type")       

    def image(self, visualization_type="sample"):
        """
        Plot the visualization based on the given visualization type.

        Parameters:
        - visualization_type (str): The type of visualization to be plotted. Default is "image".

        Returns:
        None
        """
        G = GestureDataHandler(self.datasets_dir, windows_size=self.windows_size) 
        
        while True:
            index = int(input("-1 to quit : "))

            if index == -1:
                break

            if visualization_type == "sample":
                ani = Image_Sample(index, self.model_name, G, windows_size=self.windows_size)
                if ani.generate_data():
                    ani.generate_static_plot()
            elif visualization_type == "rawdata":
                ani = Image_RawData(index, self.model_name, G, windows_size=self.windows_size)
                if ani.generate_data():
                    ani.generate_static_plot()
            elif visualization_type == "groundtruth":
                ani = Image_GroundTruth(index, self.model_name, G, windows_size=self.windows_size)
                if ani.generate_data():
                    ani.generate_static_plot()
            elif visualization_type == "slidingwindow":
                ani = Image_SlidingWindow(index, self.model_name, G, windows_size=self.windows_size)
                if ani.generate_data():
                    ani.generate_static_plot()
            # if the visualization type is not recognized, print an error message and raise an exception
            else:
                print("Invalid visualization type. Please choose from 'sample', 'rawdata', or 'groundtruth'.")
                raise ValueError("Invalid visualization type")        
    
    def realtime(self):
        """
        Starts the real-time visualization of gesture detection.

        This method initializes a serial connection, starts the animation, and then closes the serial connection.

        Returns:
            None
        """
        S = serialUSB()
        S.readSerialStart()

        ani = Ani_Realtime(self.model_name, S, windows_size=self.windows_size)
        ani.start_animation()
        S.close()

    def __str__(self):
        return f"Visualization object for model: {self.model_name} and datasets directory: {self.datasets_dir}"

        
        
        