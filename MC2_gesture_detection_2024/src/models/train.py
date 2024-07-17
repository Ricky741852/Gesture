import time
from collections import Counter
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf

from src.models.architecture import GestureModel
from src.data.preprocess import *

# if shows "find font: Font family [u'Times New Roman'] not found"
# rm ~/.cache/matplotlib -rf
plt.rcParams["font.family"] = "Times New Roman"
timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))

class GestureTensorflow:
    def __init__(
        self,
        model_path,
        save_path,
        windows_size,
        class_num,
        epoch_num,
    ):
        self.model_path = model_path
        self.curve_path = "output/curves"
        self.save_path = save_path
        self.windows_size = windows_size
        self.class_num = class_num
        self.epoch_num = epoch_num

        self.model = None

    def build_model(self):
        """Select which model you want to use for training,
        different models are defined in the model_structure.py file.
        data_preprocess function is defined in pre-process.py"""

        self.model = GestureModel(
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
        log_dir = 'logs/training_logs' + '/' + model_name
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
            train_dataY['y'],
            epochs=self.epoch_num,
            # 如果要一次處理更多的測試筆，需要將batch_size設置得較大，以確保模型能夠有效地處理這麼多的輸入。
            batch_size=64,
            validation_split=0.03,
            shuffle=True,
            callbacks=my_callbacks,
        )

        self.plot_train_history(history, model_name)

        # This is another model saving way, just in case
        # self.model.save(os.path.join('saved_model_pb', model_name))

    def plot_train_history(self, history, model_name):
        """Plotting the train and validation losses"""

        # Plot the loss history
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.legend()
        plt.title('Train and Validation Loss', fontweight='bold')

        # Save the plot of loss history
        savefig_name = os.path.join(
            self.curve_path,
            f'{model_name}_{self.class_num}_Gesture_{self.epoch_num}_epoch.png',
        )
        plt.savefig(savefig_name)
        print(
            Color.H_OK + f'Successfully saved loss plot at {savefig_name}' + Color.RESET
        )
        plt.show()

    def test_model(self, model_evaluation, index=0, test_all=False, plot_cm=False):
        """This is the sliding-window version for all the test data prediction,
        where only one gesture txt file is inferred at a time, do all 72 txt files.

        Args:
            model_evaluation: the model used to load and do model prediction
            single_idx: if you only want to test single gesture file, give a file index
            test_all: if you want to test all 72 txt gesture files, turn it to True
            plot_cm: plot a confusion matrix if true
        """

        model_path = os.path.join(os.path.dirname(__file__), '..', '..', 'output', 'models', f"{model_evaluation}.h5")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file does not exist at: {model_path}")
        
        # model_evaluation = os.path.normpath(
        #     os.path.join(self.model_path, model_evaluation)
        # )
        tag = 'without'

        self.model = load_model(model_path)
        self.model.summary()

        answer_label_list, output_label_list = [], []
        correct = 0

        if test_all:
            file_start, file_end = 0, 200
        else:
            file_start, file_end = index, index + 1

        for idx in range(file_start, file_end):
            """Each piece of txt test data belongs to an individual index"""

            # Load testing data
            test_data = test_preprocess(windows_size=self.windows_size, index=idx)
            data, answer = test_data['x'], test_data['gesture_class']

            # 因為模型的輸入是三維的(None, 50, 5)，所以需要對輸入的數據進行reshape
            reshaped_data = data.reshape(1, data.shape[0], data.shape[1])

            prediction = self.model.predict(reshaped_data, verbose=0)
            predict_gesture = np.argmax(prediction)

            print(Color.H_OK + f'Gesture answer: {answer}', end=" " + Color.RESET)
            print(Color.H_MSG + f'Gesture predict: {predict_gesture}' + Color.RESET)
            answer_label_list.append(answer)
            output_label_list.append(predict_gesture)

            if int(answer) == predict_gesture:
                correct += 1
                print(Color.H_OK + f'Predict the gesture correctly' + Color.RESET)
            else:
                print(Color.H_FAIL + f'Predict the gesture incorrectly' + Color.RESET)

        if test_all:
            # Calculate average data length of 72 test gesture files
            print(f'Windows size: {self.windows_size} (Sampling rate: 50/s)')

            # Calculate accuracy of 72 test gesture files
            accuracy = correct / 200
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

    def plot_confusion_matrix(self, y_true, y_pred):
        """Calculate confusion matrix, precision and recall"""

        cm_dir = 'output/confusion_matrix'
        check_directories(cm_dir)

        # Generate classification report of precision and recall
        class_list = [
            "Gesture " + str(i) for i in range(self.class_num)
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

        plt.xticks(range(self.class_num), class_list, fontsize=10)
        plt.yticks(range(self.class_num), class_list, fontsize=12)
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

if __name__ == "__main__":
    pass
