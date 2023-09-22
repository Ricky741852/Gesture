from matplotlib import colors
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

from tensorflow.keras.models import load_model
import numpy as np
import random
import os
import datetime
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from process import Gesture_Data
from loss import focal_loss, Index_L1_loss

from test import Gesture_Detection
from utity import cal_IoU

"""plot offline detection in animation """
class Ani_plot():
    def __init__(self, index, model, G, windows_size=128):
        self.windows_size = windows_size
        self.threshold = 0.5
        self.g_data, self.g_label, self.g_truth, self.g_class_list, self.g_class_name = G.generate_test_data(index)

        # temp
        self.g_class_list = ['1', '2', '3', '4', '5']
        self.plot_color = ["orange", "blue", "red", "brown", "green"]

        self.model = model
        self.Gesture_model = Gesture_Detection(model, windows_size=windows_size)
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(15, 15))

        # raw data plot
        self.raw_data_plot = {'data': list(), 'gbound': dict(), 'pbound': dict()}
        self.prd_data_plot = {'data': dict(), 'gbound': dict(), 'pbound': dict()}
        self.ax1.set_ylim(-90, 400)
        self.ax1.set_xlim(0, self.windows_size)
        # raw data
        for i in range(3):
            self.raw_data_plot['data'].append(self.ax1.plot([], [])[0])

        # bound
        bound = {'mid': '-', 'left': '--', 'right': '--'}
        for side, style in bound.items():
            self.raw_data_plot['pbound'][side] = self.ax1.axvline(self.windows_size, linestyle=style, color='r')
            self.raw_data_plot['gbound'][side] = self.ax1.axvline(self.windows_size, linestyle=style, color='g')
            self.prd_data_plot['pbound'][side] = self.ax2.axvline(self.windows_size, linestyle=style, color='r')
            self.prd_data_plot['gbound'][side] = self.ax2.axvline(self.windows_size, linestyle=style, color='g')

        # pridict data plot
        self.ax2.set_ylim(0, 1.3)
        self.ax2.set_xlim(0, self.windows_size)
        self.ax2.axhline(0.5, color='b')
        self.prd_data_plot['data']['predict_hm'] = list()
        for class_name, c in zip(self.g_class_list, self.plot_color):
            self.prd_data_plot['data']['predict_hm'].append(self.ax2.plot([], [], label=class_name, color=c)[0])
        self.ax2.legend(bbox_to_anchor=(0.8, 0.8, 0.3, 0.2), loc='upper right')
        self.prd_data_plot['data']['ground_truth_hm'] = self.ax2.plot([], [])[0]
        self.IoU_text = self.ax2.text(.5, .5, '', fontsize=15)

        self.plot_x = np.arange(self.windows_size)
        self.predict_data = list()
        self.AP_list = list()

    def gen_predict_data(self):

        if len(self.g_data) < self.windows_size:
            print("Data less than {}".format(self.windows_size))
            return False
        print("gen_predict_data start")
        for i in range(len(self.g_data) - self.windows_size):
            g_p = np.array(self.g_data[i:i + self.windows_size])
            g_t = np.array(self.g_truth[i:i + self.windows_size])

            # print(g_p.shape)
            # assert False

            result_predict = self.Gesture_model.predict_2(g_p)

            result_predict['raw'] = g_p
            result_predict['gtruth'] = g_t
            result_predict['gtruth_bound'] = dict()
            mid = np.argmax(g_t)

            radius = 0
            for j in range(0, len(self.g_label) - 1, 2):
                r = (self.g_label[j + 1] + self.g_label[j]) / 2
                if r > i and r < i + self.windows_size:
                    radius = (self.g_label[j + 1] - self.g_label[j]) / 2
                    # print(radius)
                    break
                # if self.g_label[j] > i  \
                #     and  self.g_label[j] < i+self.windows_size \
                #     and  self.g_label[j+1] > i  \
                #     and  self.g_label[j+1] < i+self.windows_size:

                #     radius = (self.g_label[j+1] - self.g_label[j])/2
                #     # print(radius)
                #     break

            if mid < self.windows_size:
                result_predict['gtruth_bound']['mid'] = mid
                result_predict['gtruth_bound']['left'] = mid - radius
                result_predict['gtruth_bound']['right'] = mid + radius

            result_predict['pred_bound'] = dict()
            if result_predict['hm_max_value'] >= self.threshold:
                # print(result_predict['hm_max_value'])
                result_predict['pred_bound']['mid'] = result_predict['hm_max']
                result_predict['pred_bound']['left'] = result_predict['hm_max'] - result_predict['wh']
                result_predict['pred_bound']['right'] = result_predict['hm_max'] + result_predict['wh']

            if result_predict['pred_bound'] and result_predict['gtruth_bound']:
                result_predict['IoU'] = cal_IoU(
                    [result_predict['gtruth_bound']['left'], result_predict['gtruth_bound']['right']]
                    , [result_predict['pred_bound']['left'], result_predict['pred_bound']['right']])

            self.predict_data.append(result_predict)
        print("gen_predict_data finish")

        return True

    def animate(self, i):

        predict_data = self.predict_data[i]
        # print(i)
        # raw data
        for line, data in zip(self.raw_data_plot['data'], predict_data['raw'].T):
            line.set_data(self.plot_x, data)
        # bound
        bound = {'mid': '-', 'left': '--', 'right': '--'}
        gvline = self.raw_data_plot['gbound']
        pvline = self.raw_data_plot['pbound']
        for side, style in bound.items():
            # label bound
            # if predict_data['gtruth_bound'][side] == -1:
            #     gvline[side].set_visible(False)
            # else:
            #     gvline[side].set_visible(True)
            #     gvline[side].set_xdata(predict_data['gtruth_bound'][side])
            if predict_data['gtruth_bound']:
                gvline[side].set_visible(True)
                gvline[side].set_xdata(predict_data['gtruth_bound'][side])
            else:
                gvline[side].set_visible(False)

            # predict bound
            # if predict_data['hm_max_value'] >= self.threshold:
            #     pvline[side].set_visible(True)
            #     val = predict_data['hm_max']
            #     if side == 'left':
            #         val -= predict_data['wh']
            #     elif side == 'right':
            #         val += predict_data['wh']
            #     pvline[side].set_xdata(val)
            # else:
            #     pvline[side].set_visible(False)
            if predict_data['pred_bound']:
                pvline[side].set_visible(True)
                pvline[side].set_xdata(predict_data['pred_bound'][side])
            else:
                pvline[side].set_visible(False)

        # predict data
        if 'IoU' in predict_data:
            self.IoU_text.set_text(f"i={i}\nIoU={predict_data['IoU']}\npredict gesture : {predict_data['class'] + 1}")
        else:
            self.IoU_text.set_text(f"i={i}\npredict gesture : {predict_data['class'] + 1}")

        p_gvline = self.prd_data_plot['gbound']
        p_pvline = self.prd_data_plot['pbound']
        for i in range(len(self.g_class_list)):
            self.prd_data_plot['data']['predict_hm'][i].set_data(self.plot_x, predict_data['hm'][i])
        self.prd_data_plot['data']['ground_truth_hm'].set_data(self.plot_x, predict_data['gtruth'])
        for side, style in bound.items():
            # label bound
            # if predict_data['gtruth_bound'][side] == -1:
            #     p_gvline[side].set_visible(False)
            # else:
            #     p_gvline[side].set_visible(True)
            #     p_gvline[side].set_xdata(predict_data['gtruth_bound'][side])
            if predict_data['gtruth_bound']:
                p_gvline[side].set_visible(True)
                p_gvline[side].set_xdata(predict_data['gtruth_bound'][side])
            else:
                p_gvline[side].set_visible(False)

            # predict bound
            # val = predict_data['hm_max']
            # if side == 'left':
            #     val -= predict_data['wh']
            # elif side == 'right':
            #     val += predict_data['wh']
            # p_pvline[side].set_xdata(val)
            if predict_data['pred_bound']:
                p_pvline[side].set_visible(True)
                p_pvline[side].set_xdata(predict_data['pred_bound'][side])
            else:
                p_pvline[side].set_visible(False)

    def start_animation(self):
        frames = len(self.g_data) - self.windows_size
        print("start")
        ani = animation.FuncAnimation(self.fig, self.animate, frames=frames, interval=50, repeat=True)
        plt.show()
        ani.save('./testAnimation/myAnimation.gif', writer='imagemagick', fps=20)
        print("end")

    def gen_conclusion_plot(self):
        plt.clf()

        # raw data
        plt.subplot(211)
        plt.ylabel("sensor value")
        plt.xlabel("time")
        plt.ylim(-90, 400)
        plt.xlim(0, len(self.g_data))
        plt.plot(self.g_data)
        for g in self.g_label:
            plt.axvline(g, linestyle='--')

        # predict data
        plt.subplot(212)
        plt.ylabel("IoU")
        plt.xlabel("time")
        x, y = [[] for _ in range(len(self.g_class_list))], [[] for _ in range(len(self.g_class_list))]
        for i, pred in enumerate(self.predict_data):
            if pred['hm_max'] >= self.threshold and 'IoU' in pred:
                if pred['IoU'] != 0:
                    # print(int(pred['class']),pred['hm_max']+i)
                    x[int(pred['class'])].append(pred['hm_max'] + i)
                    y[int(pred['class'])].append(pred['IoU'])
                    # print(x,y)
                    # input()
        plt.xlim(0, len(self.g_data))
        plt.ylim(-0.2, 1.2)
        # plt.stem(x,y,markerfmt='.')
        # print(x,y)
        for i, (x1, y1, color) in enumerate(zip(x, y, self.plot_color)):
            print(color)
            plt.scatter(x1, y1, s=5, label=i + 1, c=color)
        plt.legend(bbox_to_anchor=(0.8, 0.8, 0.3, 0.2), loc='upper right')

        for g in self.g_label:
            plt.axvline(g, linestyle='--')

        plt.tight_layout()
        plt.savefig(f'./conclusion_plot/{self.model}_conclusion.png')
        plt.show()


if __name__ == "__main__":

    windows_size = 100
    model_name = "Unet_classify_win_100_wh_0.1_focal_ahpha2_beta4_2022-04-13-15-08-52"
    # model_name = "pdf_mid_multi_classes_mse_epochs_20_wh_0.5_2021-02-22-20-32-33"

    # model = load_model(f"./{model_name}.h5",custom_objects={'FocalLoss_fix':FocalLoss()})
    # model = load_model(f"./{model_name}.h5",custom_objects={'focal_loss':focal_loss,'Index_L1_loss':Index_L1_loss})
    # model = load_model(f"./{model_name}.h5")
    # G = Gesture_Data(r"./testData_2021-05-03/Jack/2021-05-03-TR")
    G = Gesture_Data(r"./testData", windows_size=windows_size)
    while 1:
        index = int(input("-1 to quit : "))
        if index == -1:
            break

        ani = Ani_plot(index, model_name, G, windows_size=windows_size)
        if ani.gen_predict_data():
            ani.start_animation()
            ani.gen_conclusion_plot()
            # pass

    # windows_size = 100
    # threshold = 0.8
    # g_data , g_label ,g_truth , g_class_list,g_class_name = G.generate_test_data(1)

    # # g_truth = np.array(g_truth)

    # fig, (ax1, ax2, ax3) = plt.subplots(3,1)
    # for ax in [ax1, ax2, ax3]:
    #     ax.set_xlim([0, 100])

    # x = np.arange(0, windows_size,1)
    # print(x)
    # raw_data_line, = ax1.plot([],[])
    # ground_truth_line, = ax2.plot([],[])
    # predict_line, = ax3.plot([],[])

    # def animate(i):
    #     # g_p = np.array(g_data[i:i+windows_size])
    #     g_t = np.array(g_truth[i:i+windows_size])

    #     # raw_result = model.predict(g_p.reshape(1,100,3))
    #     # maxpool_hm = tf.nn.max_pool1d(input = raw_result[0],ksize = 3, strides=1,padding='SAME')
    #     # maxpool_hm = maxpool_hm.numpy().reshape(100)

    #     print(g_t.shape)
    #     # print(x.shape)

    #     # raw_data_line.set_data(x,g_p)
    #     ground_truth_line.set_ydata(g_t)
    #     # predict_line.set_data(x,maxpool_hm)
    #     print(i)
    #     return ground_truth_line
    #     # return raw_data_line#,ground_truth_line,predict_line

    # frames = len(g_data)-windows_size
    # print("start")
    # ani = animation.FuncAnimation(fig, animate, frames=frames, interval=50)

    # plt.show()
