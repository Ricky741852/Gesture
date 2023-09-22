# import tensorflow as tf
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#     except RuntimeError as e:
#         print(e)

# from tensorflow.keras.models import load_model
import numpy as np
import random
import os
import datetime
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from process import Gesture_Data
from loss import focal_loss,Index_L1_loss

from test import Gesture_Detection
from serial_usb import serialUSB


class Ani_plot():
    def __init__(self,model,SerialData,windows_size=100):
        self.windows_size = windows_size
        self.threshold = 0.35
        
        self.SerialData = SerialData

        self.model = model
        self.Gesture_model = Gesture_Detection(model,windows_size=windows_size) 
        self.fig, (self.ax1, self.ax2) = plt.subplots(2,1,figsize=(15,15))

        self.g_class_list = ['1','2','3','4','5']
        self.plot_color = ["orange","blue","red","brown","green"]

        # raw data plot
        self.raw_data_plot = {'data':list(),'pbound':dict()}
        self.prd_data_plot = {'data':dict(),'pbound':dict()}
        self.ax1.set_ylim(-100,400)
        self.ax1.set_xlim(0,self.windows_size)
        self.FPS_text = self.ax1.text(.5, .5, '', fontsize=15)

        # raw data
        for _ in range(3):
            self.raw_data_plot['data'].append(self.ax1.plot([],[])[0])
        
        # bound
        bound = {'mid':'-','left':'--','right':'--'}
        for side,style in bound.items():
            self.raw_data_plot['pbound'][side]   = self.ax1.axvline(self.windows_size,linestyle=style,color='r')
            # self.raw_data_plot['gbound'][side]   = self.ax1.axvline(self.windows_size,linestyle=style,color='g')
            self.prd_data_plot['pbound'][side]   = self.ax2.axvline(self.windows_size,linestyle=style,color='r')
            # self.prd_data_plot['gbound'][side]   = self.ax2.axvline(self.windows_size,linestyle=style,color='g')

        # pridict data plot
        self.ax2.set_ylim(0,1)
        self.ax2.set_xlim(0,self.windows_size)
        self.prd_data_plot['data']['predict_hm'] = list()
        for class_name,c in zip(self.g_class_list,self.plot_color):
            self.prd_data_plot['data']['predict_hm'].append(self.ax2.plot([],[],label=class_name,color=c)[0])
        
        # self.prd_data_plot['data']['ground_truth_hm'] = self.ax2.plot([],[])[0]

        self.plot_x = np.arange(self.windows_size)
        # self.predict_data = None
        self.previousTimer = 0
    
    def gen_predict_data(self,data):
        
        g_p = np.array(data).T
        # print(g_p.shape)
        result_predict = self.Gesture_model.predict_2(g_p)

        return result_predict
    
    def init_data(self):
        self.SerialData.getSerialData()
        self.SerialData.data
        for line,data in zip(self.raw_data_plot['data'],self.SerialData.data):
            # print(data.shape)
            line.set_data(self.plot_x,data)
        return tuple(self.raw_data_plot['data'])

        # return self.raw_data_plot['data'][0] ,self.raw_data_plot['data'][1] ,self.raw_data_plot['data'][2] 

    def get_data(self):
        while True:
            data = self.SerialData.getSerialData()
            predict_data = self.gen_predict_data(data)
            # predict_data = dict()
            # predict_data['hm'] = self.SerialData.data[0]
            yield self.SerialData.data,predict_data

    def animate(self,data):
        # print("TTTTTTTTTTTTTTTTTTTTTT")
        rawdata , predict_data = data
        #bound 
        bound = {'mid':'-','left':'--','right':'--'}
        # gvline = self.raw_data_plot['gbound']
        

        # predict_data = self.gen_predict_data(SerialData.data)
        # print(predict_data)
        # if self.predict_data is not None:
        #     predict_data = self.predict_data
        # else:
        #     print("No data in predict_data")
        #     return False

        currentTimer = time.perf_counter()
        # raw data
        for line,data in zip(self.raw_data_plot['data'],rawdata):
            # print(data.shape)
            line.set_data(self.plot_x,data)
        # print(predict_data['hm'].shape)
        # print(predict_data['hm'])
        # print(len(self.prd_data_plot['data']['predict_hm']))
        for i in range(len(self.g_class_list)):
            self.prd_data_plot['data']['predict_hm'][i].set_data(self.plot_x,predict_data['hm'][i])
        
        end_time =  time.perf_counter()
        print('total {0:.7f}'.format(end_time - self.previousTimer))
        if predict_data['class'] != 0:
            self.FPS_text.set_text("FPS = {:.0f} , Predict gesture : {}".format(1/(end_time - self.previousTimer) , predict_data['class']+1 ))
        else:
            self.FPS_text.set_text("FPS = {:.0f} ".format(1/(end_time - self.previousTimer)))
        
        self.previousTimer = end_time

        # p_gvline = self.prd_data_plot['gbound']
        p_pvline = self.prd_data_plot['pbound']
        pvline = self.raw_data_plot['pbound']
        
        predict_data['pred_bound'] = dict()
        if predict_data['hm_max_value'] >= self.threshold:
            # print(result_predict['hm_max_value'])
            predict_data['pred_bound']['mid'] = predict_data['hm_max']
            predict_data['pred_bound']['left'] = predict_data['hm_max']-predict_data['wh']
            predict_data['pred_bound']['right'] =  predict_data['hm_max']+predict_data['wh']
        

        for side,style in bound.items():
            # label bound
            # if predict_data['gtruth_bound']:
            #     p_gvline[side].set_visible(True)
            #     p_gvline[side].set_xdata(predict_data['gtruth_bound'][side])
            # else:
            #     p_gvline[side].set_visible(False)

            # predict bound
            if predict_data['pred_bound']:
                p_pvline[side].set_visible(True)
                p_pvline[side].set_xdata(predict_data['pred_bound'][side])
                pvline[side].set_visible(True)
                pvline[side].set_xdata(predict_data['pred_bound'][side])
            else:
                p_pvline[side].set_visible(False)
                pvline[side].set_visible(False)



        return tuple(self.raw_data_plot['data']) \
                + tuple(self.prd_data_plot['data']['predict_hm'])\
                + (self.FPS_text,)\
                + tuple(self.prd_data_plot['pbound'].values())\
                + tuple(self.raw_data_plot['pbound'].values())


        #bound 
        # bound = {'mid':'-','left':'--','right':'--'}
        # # gvline = self.raw_data_plot['gbound']
        # pvline = self.raw_data_plot['pbound']
        # for side,style in bound.items():
            
        #     if predict_data['hm_max_value'] >= self.threshold:
        #         pvline[side].set_visible(True)
        #         val = predict_data['hm_max']
        #         if side == 'left':
        #             val -= predict_data['wh']
        #         elif side == 'right':
        #             val += predict_data['wh']
        #         pvline[side].set_xdata(val)
        #     else:
        #         pvline[side].set_visible(False)

        # print("bound-----------------------------------------------------")
        # predict data
        # p_gvline = self.prd_data_plot['gbound']

        # p_pvline = self.prd_data_plot['pbound']
        # self.prd_data_plot['data']['predict_hm'].set_data(self.plot_x,predict_data['hm'])

        # self.prd_data_plot['data']['ground_truth_hm'].set_data(self.plot_x,predict_data['gtruth'])
        # for side,style in bound.items():
        #     # label bound
        #     # if predict_data['gtruth_bound'][side] == -1:
        #     #     p_gvline[side].set_visible(False)
        #     # else:
        #     #     p_gvline[side].set_visible(True)
        #     #     p_gvline[side].set_xdata(predict_data['gtruth_bound'][side])

        #     # predict bound
        #     val = predict_data['hm_max']
        #     if side == 'left':
        #         val -= predict_data['wh']
        #     elif side == 'right':
        #         val += predict_data['wh']
        #     p_pvline[side].set_xdata(val)

        # print("END-----------------------------------------------------")
    

        
    def start_animation(self):
        print("start")
        # ani = animation.FuncAnimation(self.fig, self.animate,fargs=(SerialData,1), interval=1)

        ani = animation.FuncAnimation(fig = self.fig
                                    ,func = self.animate
                                    ,frames= self.get_data
                                    ,init_func= self.init_data
                                    , interval=0
                                    , blit=True)
        plt.show()
        print("end")
        







if __name__ == "__main__":
    
    
    # model_name = "Unet_classify_win_100_wh_0.1_focal_ahpha2_beta4_2021-06-15-06-47-40"
    model_name = "Unet_classify_win_100_wh_0.1_focal_ahpha2_beta4_2021-08-16-15-13-56"
    
    windows_size = 100
    portName = '/dev/ttyACM1'     # for windows users
    baudRate = 115200
    maxPlotLength = windows_size
    dataNumBytes = 4        # number of bytes of 1 data point
    dataInNum = 3
    seiral = serialUSB(portName, baudRate, maxPlotLength, dataNumBytes,dataInNum)   # initializes all required variables
    seiral.readSerialStart()                                               # starts background thread

    ani = Ani_plot(model_name,seiral,windows_size=windows_size)

    ani.start_animation()

    seiral.close()
    


    

    




