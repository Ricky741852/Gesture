# Utils Module

This module contains utility scripts used across the project.

- `data_helpers/`: Contains scripts for adjusting raw data.
- `spotting_GUI/`: Contains scripts for extracting data from folders and visualizing data for marking gesture start and end points.
- `gaussian_groundtruth.py`: Generates ground truth using a Gaussian-like kernel function.
- `random_edit_distance.py`: Generates random combinations of continuous gestures for testing.
- `receive_data.py`: Receives gesture data through `serial_usb.py` and processes it into .txt files.
- `serial_usb.py`: Handles serial port connections and background threads for continuous data collection.
- `library.py`: Path and file confirmation, as well as color information for console outputs.
