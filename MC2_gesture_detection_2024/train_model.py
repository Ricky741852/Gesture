import time
from collections import Counter
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf

import quantize_model
from model_stucture import RevisedModel
from pre_process import *
from quantize_model import quantize_signals_windows

# if shows "find font: Font family [u'Times New Roman'] not found"
# rm ~/.cache/matplotlib -rf
plt.rcParams["font.family"] = "Times New Roman"
timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))

class GestureTensorflow:
    def __init__(
        self,
        model_path,
        save_path,
        include_path,
        windows_size,
        windows_stride,
        class_num,
        epoch_num,
    ):
        self.model_path = model_path
        self.save_path = save_path
        self.include_path = include_path
        self.windows_size = windows_size
        self.windows_stride = windows_stride
        self.class_num = class_num
        self.epoch_num = epoch_num

        self.model = None

    def build_model(self):
        """Select which model you want to use for training,
        different models are defined in the model_structure.py file.
        data_preprocess function is defined in pre-process.py"""

        self.model = RevisedModel(
            5, class_num=self.class_num, windows_size=self.windows_size
        ).build_model()

        if not os.path.isfile(os.path.join(self.save_path, f'format_train_data.pt')):
            data_preprocess(self.windows_size)

    def load_data_from_file(self, tag):
        data = torch.load(os.path.join(self.save_path, f'format_{tag}_data.pt'))
        dataX = data['dataX']
        dataY = data['dataY']
        print(f"Length of {tag} data: {len(dataX['x'])}")
        return dataX, dataY

    def train_model(self):
        """Training our model"""

        # Load training data
        train_dataX, train_dataY = self.load_data_from_file(tag='train')

        # Set up the model's name and logging directories
        model_name = f"model_{timestamp}"
        log_dir = 'training_logs'
        check_directories(log_dir)

        # Set up the model callbacks during training
        my_callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=30, monitor='val_loss'),
            tf.keras.callbacks.TensorBoard(
                log_dir=log_dir,
                histogram_freq=1,
                write_graph=True,
                profile_batch=0,
                embeddings_freq=1,
                embeddings_metadata='metadata.tsv',
                update_freq='batch',
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(self.model_path, f'{model_name}.h5'),
                monitor='val_loss',
                mode='min',
                # save_weights_only=True,
                save_best_only=True,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.1,
                patience=5,
                verbose=1,
                min_delta=1e-4,
                mode='auto',
            ),
        ]

        # Start training model
        history = self.model.fit(
            train_dataX['x'],
            train_dataY['hm'],
            epochs=self.epoch_num,
            # 如果要一次處理更多的測試筆，需要將batch_size設置得較大，以確保模型能夠有效地處理這麼多的輸入。
            batch_size=64,
            validation_split=0.03,
            shuffle=True,
            callbacks=my_callbacks,
        )

        self.plot_train_history(history, model_name)

        # This is another model saving way, just in case
        self.model.save(os.path.join('saved_model_pb', model_name))

    def plot_train_history(self, history, model_name):
        """Plotting the train and validation losses"""

        # Plot the loss history
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.legend()
        plt.title('Train and Validation Loss', fontweight='bold')

        # Save the plot of loss history
        savefig_name = os.path.join(
            self.model_path,
            f'{model_name}_train_and_val_loss_{self.epoch_num}_epoch.png',
        )
        plt.savefig(savefig_name)
        print(
            Color.H_OK + f'Successfully saved loss plot at {savefig_name}' + Color.RESET
        )
        plt.show()

    def test_model(self, model_evaluation, single_idx=0, test_all=False, plot_cm=False, post_process=False):
        """This is the sliding-window version for all the test data prediction,
        where only one gesture txt file is inferred at a time, do all 72 txt files.

        Args:
            model_evaluation: the model used to load and do model prediction
            single_idx: if you only want to test single gesture file, give a file index
            test_all: if you want to test all 72 txt gesture files, turn it to True
            plot_cm: plot a confusion matrix if true
        """

        model_evaluation = os.path.normpath(
            os.path.join(self.model_path, model_evaluation)
        )
        tag = 'without'

        self.model = load_model(model_evaluation)
        self.model.summary()

        answer_label_list, output_label_list, data_length_list, windows_num_list = [], [], [], []
        correct = 0

        if test_all:
            file_start, file_end = 0, 300
        else:
            file_start, file_end = single_idx, single_idx + 1

        for idx in range(file_start, file_end):
            """Each piece of txt test data belongs to an individual index"""

            # Load testing data
            my_test_data = test_preprocess(windows_size=self.windows_size, index=idx)
            data, answer = my_test_data['x'], my_test_data['gesture_class']
            print(f'Data length: {len(data)}')
            data_length_list.append(len(data))  # Record the length of each test data for calculation of average length

            windows = self.make_sliding_windows(data)
            windows_num_list.append(windows.shape[0])

            predictions = self.model.predict(windows, verbose=0)
            predict_label_list = []

            for i in range(len(predictions)):
                # Check which heatmap index is the max value in
                output = torch.from_numpy(predictions[i])
                max_value = torch.max(output)
                max_value_index = torch.argmax(output)
                predict_label = int((max_value_index % 6) + 1)
                predict_label_list.append(predict_label)

                # Post-process with threshold for prediction output
                if post_process:
                    # The threshold value is obtained by mode = 'get_threshold', get heatmaps of data labeled as zero
                    # 資料集中的手勢資料和背景資料在特徵上有明顯的區別，使得模型可以較容易地區分它們。另外，手勢資料的數量也可能起到了作用。如果手勢資料的數量相對較多，而背景資料的數量相對較少，那麼模型可能更容易學習如何區分這兩類。
                    thresholds = {1: 0.5862, 2: 0.7375, 3: 0.7749, 4: 0.6902, 5: 0.6737,6: 0.7698}
                    tag = 'with'
                    if predict_label in thresholds:
                        if max_value > thresholds[predict_label]:
                            predict_label_list.append(predict_label)
                        else:
                            predict_label = 0
                            predict_label_list.append(predict_label)

                print(Color.WARN + f'Gesture answer: {answer}', end=" " + Color.RESET)
                print(
                    Color.H_WARN
                    + f'Gesture window predict: {predict_label}'
                    + Color.RESET
                )
            
            # Post-process for gesture prediction
            # Record the most frequently occurring prediction result in predict_label_list, exclude zero values
            predict_label_to_count = [x for x in predict_label_list if x != 0]
            counter = Counter(predict_label_to_count)
            most_common_value = counter.most_common(1)[0][0]

            print(Color.H_OK + f'Gesture answer: {answer}', end=" " + Color.RESET)
            print(Color.H_MSG + f'Gesture predict: {most_common_value}' + Color.RESET)
            answer_label_list.append(answer)
            output_label_list.append(most_common_value)

            if int(answer) == most_common_value:
                correct += 1
                print(Color.H_OK + f'Predict the gesture correctly' + Color.RESET)
            else:
                print(Color.H_FAIL + f'Predict the gesture incorrectly' + Color.RESET)

        if test_all:
            # Calculate average data length of 72 test gesture files
            print(f'Windows size: {self.windows_size} (Sampling rate: 50/s)')
            print(f'Windows stride: {self.windows_stride} (Sliding windows)')
            average_data_length = sum(data_length_list) / len(data_length_list)
            print(
                Color.INFO
                + f'Average length of test data: {average_data_length}'
                + Color.RESET
            )
            average_windows_num = sum(windows_num_list) / len(windows_num_list)
            print(
                Color.INFO
                + f'Average windows num of test data: {average_windows_num}'
                + Color.RESET
            )

            # Calculate accuracy of 72 test gesture files
            accuracy = correct / 300
            print(
                Color.MSG
                + f'Accuracy of test data ({tag} post-processing): {accuracy}'
                + Color.RESET
            )

            if plot_cm:
                answer_label_list = list(
                    map(int, answer_label_list)
                )  # turn string to int
                self.plot_confusion_matrix(answer_label_list, output_label_list)

    def predict_scattered_test_data(self, model_evaluation, do_eval=True, do_confusion_matrix=False):
        """Caution. This function is only for debugging used, format_test_data.pt is a test set with scattered data.
        The test data in format_test_data.pt is scattered into pieces of (several, 100, 3) data.
        If you want to predict a complete gesture file, please execute test_model, but not this function.
        """

        if do_eval:
            # Load test data
            test_dataX, test_dataY = self.load_data_from_file(tag='test')

            # Load model for evaluation
            model_evaluation = os.path.normpath(
                os.path.join(self.model_path, model_evaluation)
            )
            self.model = load_model(model_evaluation)

            # run all prediction at the same time
            predict_output = self.model.predict(test_dataX['x'])

            correct = 0
            predict_label_list = []
            max_value_list_1, max_value_list_2, max_value_list_3, max_value_list_4, max_value_list_5,max_value_list_6 = [], [], [], [], [], []
            max_value_lists_dict = {
                1: max_value_list_1,
                2: max_value_list_2,
                3: max_value_list_3,
                4: max_value_list_4,
                5: max_value_list_5,
                6: max_value_list_6,
            }

            for i in range(len(predict_output)):
                output = torch.from_numpy(predict_output[i])
                # print(f'Output heatmap: {output}')
                max_value = torch.max(output)
                max_value_index = torch.argmax(output)
                predict_label = int(
                    (max_value_index % 6) + 1
                )  # check which heatmap index is the max value in

                # The threshold value is obtained by averaging the heatmaps of gesture data labeled as zero
                thresholds = {1: 0.5862, 2: 0.7375, 3: 0.7749, 4: 0.6902, 5: 0.6737,6: 0.7698}
                if predict_label in thresholds:
                    if max_value > thresholds[predict_label]:
                        predict_label_list.append(predict_label)
                    else:
                        predict_label = 0
                        predict_label_list.append(predict_label)

                if predict_label in max_value_lists_dict:
                    max_value_lists_dict[predict_label].append(max_value)

                if int(test_dataX['label'][i]) == predict_label:
                    correct = correct + 1

                print(
                    Color.H_WARN + f"Gesture answer: {test_dataX['label'][i]}",
                    end=' ' + Color.RESET,
                )
                print(Color.H_MSG + f'Gesture predict: {predict_label}' + Color.RESET)

            # Calculate accuracy rate
            incorrect = len(test_dataX['x']) - correct
            accuracy = correct / len(test_dataX['x'])
            print(Color.H_OK + f'\nAccuracy: {accuracy}' + Color.RESET)
            print(
                f'Correctly predicted gestures: {correct}\nIncorrectly predicted gestures: {incorrect}'
            )

            self.count_test_data_gesture(test_dataX['label'], tag='Answer')
            self.count_test_data_gesture(str(predict_label_list), tag='Predict')

            if do_confusion_matrix:
                true_label_list = list(
                    map(int, test_dataX['label'])
                )  # turn string to int
                self.plot_confusion_matrix(true_label_list, predict_label_list)
        else:
            return None

    def predict_zero_label_gesture(self, model_evaluation, do_eval):
        """Generate a heatmap of zero label gestures to define the threshold"""

        if not os.path.isfile(os.path.join(self.save_path, f'format_zero_data.pt')):
            data_preprocess(self.windows_size)

        if do_eval:
            # Load zero label data
            test_dataX, test_dataY = self.load_data_from_file(tag='zero')

            # Load model for evaluation
            model_evaluation = os.path.normpath(
                os.path.join(self.model_path, model_evaluation)
            )
            self.model = load_model(model_evaluation)

            # run all prediction at the same time
            predict_output = self.model.predict(test_dataX['x'])

            predict_label_list = []
            max_value_list_1, max_value_list_2, max_value_list_3, max_value_list_4, max_value_list_5, max_value_list_6 = [], [], [], [], [], []
            max_value_lists_dict = {
                1: max_value_list_1,
                2: max_value_list_2,
                3: max_value_list_3,
                4: max_value_list_4,
                5: max_value_list_5,
                6: max_value_list_6,
            }

            for i in range(len(predict_output)):
                print(
                    Color.H_WARN + f"Gesture answer: {test_dataX['label'][i]}",
                    end=' ' + Color.RESET,
                )

                output = torch.from_numpy(predict_output[i])
                max_value = torch.max(output)
                max_value_index = torch.argmax(output)
                # print(f'Max value of heatmap:{max_value}')
                predict_label = int(
                    (max_value_index % 6) + 1
                )  # check which heatmap index is the max value in
                predict_label_list.append(predict_label)

                if predict_label in max_value_lists_dict:
                    max_value_lists_dict[predict_label].append(max_value)

                print(Color.H_MSG + f'Gesture predict: {predict_label}' + Color.RESET)

            # Find Max value in each heatmap
            average_value_list = []
            print(
                Color.H_INFO
                + '\nGet heatmap threshold for post-processing'
                + '-' * 10
                + Color.RESET
            )
            for i in range(1, 7):
                print(
                    f'Max_value_list_{i}: {max(max_value_lists_dict[i]):.4f}', end='  '
                )
                print(
                    f'Avg_value_list_{i}: {sum(max_value_lists_dict[i]) / len(max_value_lists_dict[i]):.4f}'
                )
                average_value_list.append(
                    sum(max_value_lists_dict[i]) / len(max_value_lists_dict[i])
                )

        else:
            return None

    def plot_confusion_matrix(self, y_true, y_pred):
        """Calculate confusion matrix, precision and recall"""

        cm_dir = 'confusion_matrix'
        check_directories(cm_dir)

        # Generate classification report of precision and recall
        class_list = [
            "Gesture " + str(i) for i in range(1, self.class_num + 1)
        ]  # Gesture 1, ..., Gesture n
        print(Color.H_INFO + f'Classification Report ' + '=' * 60 + Color.RESET)
        print(classification_report(y_true, y_pred, target_names=class_list))

        # Save report to confusion_matrix/20230426_classification_report.txt
        report = os.path.join(cm_dir, f'{timestamp}_classification_report.txt')
        with open(report, 'w') as f:
            print(
                classification_report(y_true, y_pred, target_names=class_list), file=f
            )

        # Plot confusion matrix and save, choose seaborn or matplotlib
        confusion_mat = confusion_matrix(y_true, y_pred)

        # Plot confusion matrix with values
        plt.imshow(confusion_mat, cmap='Blues')

        # Add text annotations to each cell
        thresh = confusion_mat.max() / 2.0
        for i in range(confusion_mat.shape[0]):
            for j in range(confusion_mat.shape[1]):
                plt.text(j, i, format(confusion_mat[i, j], 'd'),
                         horizontalalignment="center",
                         color="white" if confusion_mat[i, j] > thresh else "black",
                         fontsize=10)

        plt.xticks(range(6), class_list, fontsize=10)
        plt.yticks(range(6), class_list, fontsize=12)
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold', )
        plt.xlabel('Predicted Gesture', fontsize=14, fontweight='bold')
        plt.ylabel('True Gesture', fontsize=12, fontweight='bold')
        savefig_name = f'{timestamp}_confusion_matrix.png'
        plt.savefig(os.path.join(cm_dir, savefig_name))
        plt.show()

    def count_test_data_gesture(self, label_to_match, tag):
        """Count how many of each gesture there are in list"""
        label_counter = {str(i): 0 for i in range(0, self.class_num + 1)}
        for each_label in label_to_match:
            if each_label in label_counter:
                label_counter[each_label] += 1
        print(f'Gesture {tag} Count: {label_counter}')

    def make_sliding_windows(self, data):
        """Sampling rate = 50/s, windows_stride is the step size of the sliding window
        For example: windows_size = 100, windows_stride = 50, it means that window slides per second
        """
        # Sampling rate = 50/s, windows_stride is the step size of the sliding window
        stride = self.windows_stride

        # Generate sliding windows
        windows = []
        for i in range(0, len(data) - self.windows_size + 1, stride):
            window = data[i: i + self.windows_size]
            windows.append(window)
        windows = np.array(windows)
        print(f'Window shape: {windows.shape}')

        return windows

    def get_model_layer_weights(self, model_evaluation):
        """View the weights of each conv layers in the float32 model, debug used"""

        # Load model
        model_evaluation = os.path.normpath(
            os.path.join(self.model_path, model_evaluation)
        )
        self.model = load_model(model_evaluation)
        np.set_printoptions(precision=2, suppress=True)

        layer_cnt = 1
        for layer in self.model.layers:
            if layer.get_weights():
                if 'conv1d' in layer.name:
                    conv = np.array(layer.get_weights())
                    print(
                        Color.INFO
                        + f'conv1d_{layer_cnt}.shape: {conv.shape}'
                        + Color.RESET
                    )
                    print(f'conv1d_{layer_cnt}: {conv}')
                    layer_cnt += 1
                elif 'batch_normalization' in layer.name:
                    BN = np.array(layer.get_weights())
                    print(Color.INFO + f'bn.shape: {BN.shape}' + Color.RESET)
                    print(f'bn: {BN}')

    def quantize_model(self, model_quantization):
        """This function is used to quantize the float32 model into an int8 model,
        Most of the function used here are written in quantize_model.py file.
        """

        train_featuresX, train_featuresY = self.load_data_from_file(tag='train')
        total_features = train_featuresX['x'].astype(np.float32)

        # Reshape total_features
        cal_shape = int(total_features.size / (self.windows_size * 5))
        total_features = total_features.reshape((cal_shape, 1, 1, self.windows_size, 5))
        model_path = os.path.normpath(os.path.join(self.model_path, model_quantization))
        header_name = os.path.normpath(
            os.path.join(self.include_path, f'quantized_{model_quantization[:-10]}.h')
        )

        # TODO: 20230501 fixing bug, a few of the training data will result in the following errors
        # Try to solve: zeroPoint = int(zeroPoint) # ValueError: cannot convert float NaN to integer
        # Remove the non-compliant data from the total features
        data_removed = [
            1935,
            2861,
            3031,
            3052,
            5590,
            9023,
            9032,
            13615,
            13626,
            13672,
            13673,
            13674,
        ]
        total_features = np.delete(total_features, data_removed, axis=0)
        print(f'Total_features.shape: {total_features.shape}')

        S1, Z1, S4, Z4 = quantize_model.get_layer_factor(
            window_size=self.windows_size,
            model_path=model_path,
            total_features=total_features,
        )

        quantize_model.make_header(
            window_size=self.windows_size,
            S1=S1,
            Z1=Z1,
            S4=S4,
            Z4=Z4,
            model_path=model_path,
            header_name=header_name,
        )

        # Save quantized model as quantized_model_20230426.h
        quantize_model_name = f'quantized_{model_quantization[:-10]}.h'

        return quantize_model_name

    def quantize_input(self, single_idx, write_all=False):
        """This function is used to quantize input data into an int8 datatype,
        quantize_signals_windows function is written in quantize_input.py file.

        Args:
            single_idx: if you only want to write single test data to an input hfile, for debugging used
            write_all: if you want to write all test data to hfile, turn it to True
        """

        # include_header/gesture_input/...
        input_hfile_dir = os.path.join(self.include_path, 'gesture_input')
        check_directories(input_hfile_dir)

        if write_all:
            file_start, file_end = 0, 300
        else:
            file_start, file_end = single_idx, single_idx + 1

        for idx in range(file_start, file_end):
            """Each piece of txt test data belongs to an individual index"""

            my_test_data = test_preprocess(windows_size=self.windows_size, index=idx)
            data, answer = my_test_data['x'], my_test_data['gesture_class']
            windows = self.make_sliding_windows(data)

            input_hfile_name = f'quantized_input_{idx}.h'
            input_hfile_name = os.path.join(input_hfile_dir, input_hfile_name)
            quantize_signals_windows(windows, answer, header_name=input_hfile_name)

        return input_hfile_dir

if __name__ == "__main__":
    pass
