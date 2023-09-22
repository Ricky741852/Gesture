import numpy as np

import tensorflow as tf
from tensorflow import keras
# from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Activation, GlobalAveragePooling1D, Dense, \
    Conv2DTranspose, Lambda, Add, MaxPooling1D, concatenate

from loss import binary_focal_loss, focal_loss, Index_L1_loss


def Conv1DTranspose(input_tensor, filters, kernel_size, strides=2, padding='same'):
    """
        input_tensor: tensor, with the shape (batch_size, time_steps, dims)
        filters: int, output dimension, i.e. the output tensor will have the shape of (batch_size, time_steps, filters)
        kernel_size: int, size of the convolution kernel
        strides: int, convolution step size
        padding: 'same' | 'valid'
    """
    x = Lambda(lambda x: K.expand_dims(x, axis=2))(input_tensor)
    x = Conv2DTranspose(filters=filters, kernel_size=(kernel_size, 1), strides=(strides, 1), padding=padding)(x)
    x = Lambda(lambda x: K.squeeze(x, axis=2))(x)
    return x


class SimpleNet(object):
    """
    docstring
    """

    def __init__(self, channel, class_num=1, windows_size=100):
        self.model = None
        self.channel = channel
        self.windows_size = windows_size
        self.class_num = class_num

    def _make_header_layer(self, layer, in_ch, out_ch, name):
        # header = nn.Sequential(
        #     nn.Conv2d(in_ch, in_ch, 3, 1, 1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(in_ch, out_ch, 1, 1)
        # )

        header = Conv1D(in_ch, 3, activation='relu', use_bias=False, padding='same')(layer)
        header = Conv1D(out_ch, 1, activation=None, use_bias=False, name=name)(header)

        return header

    def build_model(self):
        signal_input = Input(shape=(self.windows_size, self.channel), name="signal_input")
        out = Conv1D(128, 3, activation=None, use_bias=False, padding='same')(signal_input)
        out = BatchNormalization()(out)
        out = Activation('relu')(out)
        out = Conv1D(128, 3, strides=2, activation=None, use_bias=False)(out)
        out = BatchNormalization()(out)
        out = Activation('relu')(out)
        # out = Conv1D(128, 2, strides=2, activation=None, use_bias=False)(out)  
        # out = BatchNormalization()(out)
        # out = Activation('relu')(out)
        out = Conv1D(256, 3, strides=2, activation=None, use_bias=False)(out)
        out = BatchNormalization()(out)
        out = Activation('relu')(out)
        # out = Conv1D(256, 2, strides=2, activation=None, use_bias=False)(out)  
        # out = BatchNormalization()(out)
        # tf.keras.layers.Conv1DTranspose
        out = Conv1DTranspose(out, 256, 3)
        out = BatchNormalization()(out)
        out = Activation('relu')(out)
        out = Conv1DTranspose(out, 128, 3)
        out = BatchNormalization()(out)
        out = Activation('relu')(out)

        # out_hm = Conv1D(256,2, activation='relu', use_bias=False)(out)
        # out_hm = Conv1D(self.class_num,1, activation='relu', use_bias=False,name="hm")(out_hm)

        # out_wh = Conv1D(256,2, activation='relu', use_bias=False)(out)
        # out_wh = Conv1D(1,1, activation='relu', use_bias=False,name="wh")(out_wh)
        out_hm = self._make_header_layer(out, 256, self.class_num, "hm")
        out_wh = self._make_header_layer(out, 256, 1, "wh")

        model = keras.Model(
            inputs=[signal_input],
            outputs=[out_hm, out_wh]
        )
        self.model = model

        model.summary()

        model.compile(
            optimizer=keras.optimizers.Adam(),
            loss={
                # "hm":binary_focal_loss(alpha=.25, gamma=2),
                # "hm":FocalLoss(),
                # "hm":keras.losses.mean_squared_error,
                "hm": focal_loss,
                # 'hm':keras.losses.categorical_crossentropy,
                # "wh":keras.losses.mean_absolute_error
                "wh": Index_L1_loss
            },
            loss_weights={
                "hm": 1.0,
                "wh": 0.5
            }
            # metrics=['accuracy']
        )

        return model


class ResidualNet(object):
    """
    docstring
    """

    def __init__(self, channel, class_num=1, windows_size=100):
        self.model = None
        self.channel = channel
        self.windows_size = windows_size
        self.class_num = class_num

    def _make_header_layer(self, layer, in_ch, out_ch, name):
        # header = nn.Sequential(
        #     nn.Conv2d(in_ch, in_ch, 3, 1, 1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(in_ch, out_ch, 1, 1)
        # )

        header = Conv1D(in_ch, 3, activation='relu', use_bias=False, padding='same')(layer)
        header = Conv1D(out_ch, 1, activation=None, use_bias=False, name=name)(header)

        return header

    def build_model(self):
        signal_input = Input(shape=(self.windows_size, self.channel), name="signal_input")
        out = Conv1D(128, 3, activation=None, use_bias=False, padding='same')(signal_input)
        out = BatchNormalization()(out)
        out = Activation('relu')(out)

        shortcut = Conv1D(256, 1, strides=4, activation=None, use_bias=False, padding='same')(out)

        fx = Conv1D(128, 3, strides=2, activation=None, use_bias=False, padding='same')(out)
        fx = BatchNormalization()(fx)
        fx = Activation('relu')(fx)

        fx = Conv1D(256, 3, strides=2, activation=None, use_bias=False, padding='same')(fx)
        out = Add()([shortcut, fx])
        out = BatchNormalization()(out)
        out = Activation('relu')(out)

        shortcut2 = Conv1DTranspose(out, 128, 1, strides=4, padding='same')

        out = Conv1DTranspose(out, 256, 3)
        out = BatchNormalization()(out)
        out = Activation('relu')(out)

        out = Conv1DTranspose(out, 128, 3)
        out = BatchNormalization()(out)
        out = Activation('relu')(out)

        out = Add()([shortcut2, out])

        # out_hm = Conv1D(256,2, activation='relu', use_bias=False)(out)
        # out_hm = Conv1D(self.class_num,1, activation='relu', use_bias=False,name="hm")(out_hm)

        # out_wh = Conv1D(256,2, activation='relu', use_bias=False)(out)
        # out_wh = Conv1D(1,1, activation='relu', use_bias=False,name="wh")(out_wh)
        out_hm = self._make_header_layer(out, 256, self.class_num, "hm")
        out_wh = self._make_header_layer(out, 256, 1, "wh")

        model = keras.Model(
            inputs=[signal_input],
            outputs=[out_hm, out_wh]
        )
        self.model = model

        model.summary()

        model.compile(
            optimizer=keras.optimizers.Adam(),
            loss={
                # "hm":binary_focal_loss(alpha=.25, gamma=2),
                # "hm":FocalLoss(),
                # "hm":keras.losses.mean_squared_error,
                "hm": focal_loss,
                # 'hm':keras.losses.categorical_crossentropy,
                # "wh":keras.losses.mean_absolute_error
                "wh": Index_L1_loss
            },
            loss_weights={
                "hm": 1.0,
                "wh": 0.1
            }
            # metrics=['accuracy']
        )

        return model


class U_net(object):
    def __init__(self, channel, class_num=1, windows_size=100, start_filter=8, kernal=3):
        self.model = None
        self.channel = channel
        self.windows_size = windows_size
        self.class_num = class_num

        self.start_filter = start_filter
        self.kernal = kernal

    def _residual_block(self, input, filter, kernal):
        out1 = Conv1D(filter, kernal, activation=None, use_bias=False, padding='same')(input)
        out = BatchNormalization()(out1)
        out = Activation('relu')(out)

        out = Conv1D(filter, kernal, activation=None, use_bias=False, padding='same')(out)
        out = Add()([out1, out])
        out = BatchNormalization()(out)
        out = Activation('relu')(out)

        return out

    def _encode_block(self, input, filter, kernal):
        conv = self._residual_block(input, filter, kernal)
        pool = MaxPooling1D(pool_size=2, strides=2, padding='valid')(conv)
        return conv, pool

    def _decode_block(self, input, concat, filter, kernal):
        out = Conv1DTranspose(input, filter, 3)
        out = concatenate([concat, out])
        out = self._residual_block(out, filter, kernal)

        return out

    def _make_header_layer(self, layer, in_ch, out_ch, name):
        # header = nn.Sequential(
        #     nn.Conv2d(in_ch, in_ch, 3, 1, 1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(in_ch, out_ch, 1, 1)
        # )

        header = Conv1D(in_ch, 3, activation='relu', use_bias=False, padding='same')(layer)
        header = Conv1D(out_ch, 1, activation=None, use_bias=False, name=name)(header)

        return header

    def build_model(self):
        signal_input = Input(shape=(self.windows_size, self.channel), name="signal_input")
        conv1, pool1 = self._encode_block(signal_input, self.start_filter * 4, self.kernal)
        conv2, pool2 = self._encode_block(pool1, self.start_filter * 8, self.kernal)
        # conv3,pool3 = self._encode_block(pool2,self.start_filter*4,self.kernal)
        # conv4,pool4 = self._encode_block(pool3,self.start_filter*8,self.kernal)

        encode5 = self._residual_block(pool2, self.start_filter * 16, self.kernal)

        # decode4 = self._decode_block(encode5,conv4,self.start_filter*8,self.kernal)
        # decode3 = self._decode_block(decode4,conv3,self.start_filter*4,self.kernal)
        decode2 = self._decode_block(encode5, conv2, self.start_filter * 8, self.kernal)
        decode1 = self._decode_block(decode2, conv1, self.start_filter * 4, self.kernal)

        out_hm = self._make_header_layer(decode1, 128, self.class_num, "hm")
        out_wh = self._make_header_layer(decode1, 128, 1, "wh")

        model = keras.Model(
            inputs=[signal_input],
            outputs=[out_hm, out_wh]
        )
        self.model = model

        model.summary()

        model.compile(
            optimizer=keras.optimizers.Adam(),
            loss={
                # "hm":binary_focal_loss(alpha=.25, gamma=2),
                # "hm":FocalLoss(),
                # "hm":keras.losses.mean_squared_error,
                "hm": focal_loss,
                # 'hm':keras.losses.categorical_crossentropy,
                # "wh":keras.losses.mean_absolute_error
                "wh": Index_L1_loss
            },
            loss_weights={
                "hm": 1.0,
                "wh": 0.1
            }
            # metrics=['accuracy']
        )

        return model


if __name__ == "__main__":
    # simple = SimpleNet(3,class_num=8,windows_size=100)
    # model = simple.build_model()

    # residual = ResidualNet(3,class_num=8,windows_size=100)
    # model = residual.build_model()

    unet = U_net(3, class_num=5, windows_size=128)
    model = unet.build_model()

    keras.utils.plot_model(model, "residual.png", show_shapes=True)