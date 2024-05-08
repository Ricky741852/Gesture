import serial  # 藍芽函式庫 pip install pyserial
import struct  # 二進制字串打包處理
import numpy
import os  # 目錄生成與檔案控制 
import time  # 抓取日期與時間
import msvcrt  # 按鍵監聽 (windows)
# import getch  # 按鍵監聽 (linux)
from serial_usb import serialUSB
# import keyboard
import sys
from select import select

###################### ######################
###      設定檔案路徑、使用者以及手勢       ###
############################################ 

# PATH = 'testData'  # 接收檔案儲存路徑 
PATH = 'trainData'
USER = "Andy"  # 資料蒐集者
DATE = time.strftime('%Y-%m-%d', time.localtime(time.time()))  # 當天日期
GESTURE = "1"  # 手勢動作
DATASET = "TR"
SENSOR_NUM = 5


# windows : 空白鍵按鍵監聽
# linux : q按鍵監聽
def hit_key():
    # if keyboard.read_key() == "q":
    #     return True
    if kbhit():  # 若偵測到鍵盤事件
        if ord(msvcrt.getch()) == 32:  # 若鍵盤事件為 空白鍵 = 32
            # press space to end recording
            return True
    return False


def kbhit():
    ''' Returns True if keyboard character was hit, False otherwise.
    '''

    return msvcrt.kbhit()


if __name__ == '__main__':

    folder = f"{PATH}/{GESTURE}"
    if not os.path.exists(folder):
        os.makedirs(folder)  # 產生檔案儲存路徑

    STOP_FLAG = False
    dataInNum = 5   # number of data input
    serialData = serialUSB()  # initializes all required variables, configurations are in config.ini
    

    # portName = 'COM8'  # for windows users
    # portName = '/dev/ttyACM1'  # for linux users
    # baudRate = 115200
    # maxPlotLength = 100
    # dataNumBytes = 2  # number of bytes of 1 data point   # not used
    # serialData = serialUSB(portName, baudRate, maxPlotLength, dataInNum)  # initializes all required variables

    try:
        data_total = 0
        while 1:
            data_total += 1
            # 等待開始
            # while not hit_key(): 
            #     SERIAL.flushInput()   # 清空 Comport's receive Buffer 
            #     print('waiting')
            print('waiting')
            while 1:
                c = msvcrt.getch()
                print(ord(c))
                if c == b'w':  # write data
                    break
                elif c == b'q':  # quit recording
                    STOP_FLAG = True
                    break

            if STOP_FLAG:
                print("program stop")
                break

            print(f"Start recording data {data_total}")

            deTime = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))  # deTime = 當前詳細時間
            with open(f"{folder}/{GESTURE}_{deTime}_{USER}_{DATASET}.txt", "w") as sensor_File:
                # 開始紀錄回傳的感測值
                serialData.serialConnection.reset_input_buffer()
                data_buffer = []    #用來暫存每組資料的緩衝區
                first_line = True   #用來標記是否為第一行
                while not hit_key():
                    byteData = serialData.serialConnection.read()
                    if byteData == b'e':
                        if(len(data_buffer) == dataInNum):
                            if not first_line:
                                print()
                            processed_data = ["{0:4d}".format(-(data - 100))  for _, data in enumerate(data_buffer)]
                            print(",".join(processed_data), end='')
                            sensor_File.write(",".join(processed_data))
                            sensor_File.write('\n')
                            first_line = False
                        data_buffer = []
                    else:
                        data = int.from_bytes(byteData, "big")
                        data_buffer.append(data)
            print()
            print("END recording data")
    finally:
        serialData.serialConnection.close()
