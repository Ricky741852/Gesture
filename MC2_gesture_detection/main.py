import time
import os
from process import Gesture_Data
from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt

from model import SimpleNet, ResidualNet,U_net


def plot_train_his(history, model_name):
    # # 绘制训练 & 验证的准确率值
    # pli.subplot(2,1,1)
    # plt.plot(history.history['acc'])
    # plt.plot(history.history['val_acc'])
    # plt.title('Model accuracy')
    # plt.ylabel('Accuracy')
    # plt.xlabel('Epoch')
    # plt.legend(['Train', 'Test'], loc='upper left')

    # 绘制训练 & 验证的损失值
    plt.subplot(3, 1, 1)
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.ylim(0, 10)
    plt.xlabel('Epoch')
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.legend(bbox_to_anchor=(0.8, 0.8, 0.3, 0.2), loc='upper right')

    plt.subplot(3, 1, 2)
    plt.title('hm loss')
    plt.ylabel('Loss')
    plt.ylim(0, 10)
    plt.xlabel('Epoch')
    plt.plot(history.history['hm_loss'], label='hm_loss')
    plt.plot(history.history['val_hm_loss'], label="val_hm_loss")
    plt.legend(bbox_to_anchor=(0.8, 0.8, 0.3, 0.2), loc='upper right')

    plt.subplot(3, 1, 3)
    plt.title('wh loss')
    plt.ylabel('Loss')
    plt.ylim(0, 10)
    plt.xlabel('Epoch')
    plt.plot(history.history['wh_loss'], label='wh_loss')
    plt.plot(history.history['val_wh_loss'], label='val_wh_loss')
    plt.legend(bbox_to_anchor=(0.8, 0.8, 0.3, 0.2), loc='upper right')

    save_path = "train_loss_plot"
    if not os.path.exists(save_path):
        os.makedirs(save_path)  # 產生檔案儲存路徑

    plt.savefig(f"{save_path}/{model_name}.png")
    plt.show()


if __name__ == "__main__":

    epochs = 50

    # get data
    gesture_data = Gesture_Data(r"./trainData",windows_size=100)
    gesture_data.generate_train_data()

    # test_data = Gesture_Data(r"./testData_2021-05-03/Jack/2021-05-03-TR")
    # test_data.generate_train_data()
    # G.plot(10)

    # get model
    # simplenet = SimpleNet(3,class_num=gesture_data.data_classes_total,windows_size=100)
    # model = simplenet.build_model()

    # resnet = ResidualNet(
    #     3, class_num=gesture_data.data_classes_total, windows_size=100)
    # model = resnet.build_model()

    unet = U_net(3,class_num=gesture_data.data_classes_total,windows_size=100)
    model = unet.build_model()

    date_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    model_name = f"Unet_classify_win_100_wh_0.1_focal_ahpha2_beta4_{date_time}"
    # callback
    my_callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=10, monitor='val_loss'),
        # tf.keras.callbacks.TensorBoard(log_dir='.\\logs'),
        tf.keras.callbacks.ModelCheckpoint(
            filepath="./saveModel/"+model_name,
            # save_weights_only=True,
            monitor='val_loss',
            mode='min',
            save_best_only=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.1, patience=5, verbose=1, min_delta=1e-4, mode='auto')
    ]

    # train
    train_history = model.fit(
        {"signal_input": gesture_data.x['x']},
        {"hm": gesture_data.y['hm'], "wh": gesture_data.y['wh']},
        epochs=epochs,
        batch_size=64,
        validation_split=0.2,
        # validation_data=({"signal_input": test_data.x['x']}, {
        #                  "hm": test_data.y['hm'], "wh": test_data.y['wh']}),
        shuffle=True,
        callbacks=my_callbacks
    )
    # print(train_history.history)

    plot_train_his(train_history, model_name)
    model.save("./saveModel/"+model_name)
