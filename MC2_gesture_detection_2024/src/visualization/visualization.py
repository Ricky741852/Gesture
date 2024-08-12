from src.visualization import *

class Visualization:
    def __init__(self, model_name, datasets_dir, window_size=50):
        self.window_size = window_size
        self.model_name = model_name
        self.datasets_dir = datasets_dir

    def animation(self, visualization_type="sample"):
        if visualization_type == "realtime":
            animation_realtime(self.model_name, window_size=self.window_size)
            exit()
        while True:
            index = int(input("-1 to quit : "))

            if index == -1:
                break        
            
            if visualization_type == "sample":
                animation_sample(index, self.model_name, self.datasets_dir, self.window_size)
            elif visualization_type == "simulation":
                animation_simulation(index, self.model_name, 'data/datasets/simulateData', self.window_size)
            elif visualization_type == "slidingwindow":
                animation_slidingwindow(index, self.datasets_dir, self.window_size)
            # if the visualization type is not recognized, print an error message and raise an exception
            else:
                print("Invalid animation type. Please choose from 'realtime', 'sample', 'simulation', or 'slidingwindow'.")
                raise ValueError("Invalid animation type")       

    def image(self, visualization_type="sample"):
        """
        Plot the visualization based on the given visualization type.

        Parameters:
        - visualization_type (str): The type of visualization to be plotted. Default is "sample".

        Returns:
        None
        """        
        while True:
            index = int(input("-1 to quit : "))

            if index == -1:
                break

            if visualization_type == "sample":
                image_sample(index, self.model_name, self.datasets_dir, window_size=self.window_size)
            elif visualization_type == "rawdata":
                image_rawdata(index, self.datasets_dir, window_size=self.window_size)
            elif visualization_type == "groundtruth":
                image_groundtruth(index, self.datasets_dir, window_size=self.window_size)
            elif visualization_type == "slidingwindow":
                image_slidingwindow(index, self.datasets_dir, window_size=self.window_size)
            # if the visualization type is not recognized, print an error message and raise an exception
            else:
                print("Invalid image type. Please choose from 'sample', 'rawdata', 'groundtruth', or 'slidingwindow'.")
                raise ValueError("Invalid image type")        

    def __str__(self):
        return f"Visualization object for model: {self.model_name} and datasets directory: {self.datasets_dir}"
