# MC2_gesture_detection
```

```



# Mainly Code

## `main.py` 
Training model use training data in `trainData` folder and save model to `saveModel` folder  as `.pd` format

## `real_time_detection.py`
Detecting gesture with our smart gloves which connect by bluetooth in real time.

* If detected , print `Detect result:` .

## `animation_plot_real_time.py`
Detecting gesture with our smart gloves which connect by bluetooth in real time.

* This will show plot for visulization , include  sensor raw data,model output result,detection result.

## `test_analyze-Copy1.ipynb`
jupyder notebook for testing data analyzation.
* Loading test data from `testData` folder 
* Ploting PRcurve and confusion matrics.

## `spotting_GUI`
For cutting the gesture visualization with training data and testing data by GUI made by tkinter.



# Utity Code

## `loss.py`
* include focal loss and L1 loss


## `model.py`
* define tensorflow sequential model


## `model_post_process.py`
* post process for model when testing and real time detecting

## `test.py`
using `model_post_process.py` to predict  model result


## `process.py` `process2.py`
* Loading gesture data 
* Generate training data and testing data with ground truth in `utity.py`

`process.py` for training 

`process2.py` for testing in `test_analyze-Copy1.ipynb`

## `utity.py`
* Calculate IoU
* Generate ground truth 


## `Receive_Data.py`
Collecting gesture sensor data to file for training or testing

## `serial_usb.py`
* connect to Bluetooth serial port 
* start a thread to keep storing data into queue

## `single_usb_plot_2.py`
For testing sensor data value whether it is normal or not with matplotlib figure  animation.


## `single_test.py`
same as  `single_usb_plot_2.py` but without animation,only print.