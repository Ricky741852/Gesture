# Gesture Recognition on Gemmini

## Usage

- `main.py`: main program that allows the user to choose whether to execute model training, model evaluation, or model quantization

### Training

```commandline
python main.py --mode train --epoch_num=50
```

### Testing

- If you want to inference all the test data, run this.

```commandline
python main.py --mode test --test_all --plot_cm
```

- If you want to inference single test data, run this with index between 0 to 79.

```commandline
python main.py --mode test --single_idx=0
```

### Writing to Header File

```commandline
python main.py --mode write_model
```

```commandline
python main.py --mode write_input
```

### Optional arguments

```commandline=0
  -h, --help            show this help message and exit
  -m MODE, --mode MODE  Mode of the program. Choose from: train, test, write_model, write_input, get_threshold, get_layer, none
  --eval_model EVAL_MODEL
                        Name of the model for evaluation. Default is model_20230426_150158.h5.
  --single_idx SINGLE_IDX   Index of the single file to test. Default is 0.
  --test_all                 Whether to test all files. Default is False.
  --plot_cm                Whether to plot confusion matrix during testing. Default is False.
  --post_process        Whether to do post-process during testing. Default is False.
  --windows_stride WINDOWS_STRIDE
                        The stride of sliding window. Default is 20.
  --epoch_num EPOCH_NUM
                        The number of epochs for training. Default is 50.
```


### Evaluate

- `confusion_matrix/`: generated when running in test mode, used to evaluate the model

### Others

- `library.py`: functions for main code, including formulas for quantization

## Preprocess

- `pre-process.py`: preprocess for raw training data labeling
- `utility.py`: functions required for data preprocessing
- `training_data/`: original raw data for training
- `testing_data/`: original raw data for testing
- `other_data/`: original raw data for zero-label gesture
- `format_data/`: preprocess output, formatted pt data file

## Model

- `model_structure.py`: used to establish the structure of the model
- `train_model.py`: used to train and evaluate the model
- `saved_model_h5/`: place to save tensorflow h5 model
- `saved_model_pb/`: place to save tensorflow pb model, just as a backup

## Quantization

- `quantize_model.py`: quantize the fp32 model, input data to int8 and write it into the header file
- `include_header/`: header file that needs to be placed in `gemmini-rocc-tests/include`
- `gemmini_c_code/`: main C used to perform model calculations in `gemmini-rocc-tests/bareMetalC`

## Include

- `func.h`: c function for running gesture_recognition_on_gemmini.c
- `gesture_input/`: all quantized gesture data for gemmini input
- `quantized_model_{timestamp}`: quantized model data for gemmini

## Memo

- `requirements.txt`: for installation reference only, modules list in conda env