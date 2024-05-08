from tensorflow.keras.layers import Conv1D, Activation, BatchNormalization,Dense,Flatten
from tensorflow import keras
from tensorflow.keras import Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import mean_squared_error
import os

class RevisedModel(object):
    def __init__(self, channel, class_num, windows_size=128):
        self.model = None
        self.channel = channel
        self.windows_size = windows_size
        self.class_num = class_num

    def build_model(self):
        signal_input = Input(shape=(self.windows_size, self.channel), name="signal_input")
        out = Conv1D(32, 3, strides=1, activation=None, use_bias=False, padding='same', name='conv1d_1')(signal_input)
        out = BatchNormalization(name='batch_normalization_1')(out)
        out = Activation('relu', name='relu_1')(out)
        # print('out1 :',out.shape)

        out = Conv1D(64, 3, strides=1, activation=None, use_bias=False, padding='same', name='conv1d_2')(out)
        out = BatchNormalization(name='batch_normalization_2')(out)
        out = Activation('relu', name='relu_2')(out)
        # print('out2 :',out.shape)

        out = Conv1D(128, 3, strides=1, activation=None, use_bias=False, padding='same', name='conv1d_3')(out)
        out = BatchNormalization(name='batch_normalization_3')(out)
        out = Activation('relu', name='relu_3')(out)
        # print('out3 :',out.shape)
        out = Conv1D(128, 3, strides=1, activation=None, use_bias=False, padding='same', name='conv1d_4')(out)
        out = BatchNormalization(name='batch_normalization_4')(out)
        out = Activation('relu', name='relu_4')(out)
        # print('out4 :',out.shape)

        out = Conv1D(self.class_num, 1, strides=1, activation=None, use_bias=False, padding='valid', name='conv1d_5')(out)
        out = BatchNormalization(name='batch_normalization_5')(out)
        out = Activation('relu', name='relu_5')(out)
        # print('out5 :',out.shape)

        out = Flatten(name='flatten_1')(out)
        # print('flatten_1 :',out.shape)

        out = Dense(self.class_num,activation='softmax',name='softmax_1')(out)

        self.model = keras.Model(
            inputs=signal_input,
            outputs=out
        )   

        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss = mean_squared_error,
        )

        self.model.summary()

        return self.model


if __name__ == "__main__":

    # Plot and save model structure, here we choose to use RevisedModel
    net = RevisedModel(5, class_num=6, windows_size=50)
    model = net.build_model()
    plot_structure = os.path.join('saved_model_h5', f'model_structure_{RevisedModel.__name__}.png')
    pass
