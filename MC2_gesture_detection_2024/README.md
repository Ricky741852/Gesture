# Gesture Recognition

## Project Structure

- `config.ini`: Contains serial connection settings including port, baudrate, dataSeriesLength, and numSensorGroups.
- `main.py`: The main entry point for the program.
- `requirement.txt`: Lists the dependencies required for the project environment.
- `data/`: Contains raw, processed, and dataset files. See [data/README.md](data/README.md) for more details.
- `logs/`: Contains logs generated during training and evaluation. See [logs/README.md](logs/README.md) for more details.
- `output/`: Contains output files such as images, animations, and trained models. See [output/README.md](output/README.md) for more details.
- `src/`: Contains source code organized by functionality. See [src/README.md](src/README.md) for more details.

## Usage

- `main.py`: main program that allows the user to choose whether to execute model training, model evaluation, etc.

### Training

```commandline
python main.py --mode train --epoch_num=50
```

### Testing

- If you want to inference all the test data, run this.

```commandline
python main.py --mode test --test_all --plot_cm
```

- If you want to inference single test data, run this with index between 0 to xx.

```commandline
python main.py --mode test --index=0
```

### Visualization

- If you want to show the real-time visualization, run this.

```commandline
python main.py --mode animation --animation_type=realtime
```

- If you want to show the sample data visualization in image, run this.

```commandline
python main.py --mode image --image_type=sample
```

- Note:

  - For --animation_type, you can also choose sample, simulation or slidingwindow.

  - For --image_type, you can also choose rawdata, groundtruth or slidingwindow.

### Receiving Data

To receive data in your program, you can use the following command:

```commandline
python main.py --mode record
```

You can configure the data source and other settings in the `config.ini` file.

### Labeling Data

To label data, you can use the following command:

```commandline
python src\utils\spotting_GUI\Spotting_GUI.py
```

### Optional arguments

```commandline=0
  -h, --help              show this help message and exit
  -m MODE, --mode         Mode of the program. 
                          Choose from: train, test, get_layer, animation, image, record
  --eval_model            Name of the model for evaluation. 
                          Default is model_20240703_130244.h5.
  --index                 Index of the single file to test. 
                          Default is 0.
  --test_all              Whether to test all files. 
                          Default is False.
  --plot_cm               Whether to plot confusion matrix during testing. 
                          Default is False.
  --epoch_num             The number of epochs for training. 
                          Default is 50.
  -a_type, --animation_type
                          Type of animation for visualization. 
                          Default is sample. 
                          Choose from: realtime, sample, simulation, slidingwindow.
  -i_type, --image_type   Type of image for visualization. 
                          Default is sample. 
                          Choose from: sample, rawdata, groundtruth, slidingwindow.

```