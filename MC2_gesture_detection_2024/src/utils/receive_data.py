import os  # 目錄生成與檔案控制 
import time  # 抓取日期與時間
import msvcrt  # 按鍵監聽 (windows)
from .serial_usb import serialUSB
import configparser

###################### ######################
###      設定檔案路徑、使用者以及手勢       ###
############################################ 

config = configparser.ConfigParser()
config.read('config.ini')
datasets = config.get('ReceiveSettings', 'datasets')
collecter = config['ReceiveSettings'].get('collecter')
gesture = config['ReceiveSettings'].get('gesture')

PATH = F'data/datasets/{datasets}'  # 接收檔案儲存路徑 
USER = collecter  # 資料蒐集者
DATE = time.strftime('%Y-%m-%d', time.localtime(time.time()))  # 當天日期
GESTURE = gesture  # 手勢動作
SENSOR_NUM = 5

# windows : 空白鍵按鍵監聽
# linux : q按鍵監聽
def _hit_key():
    if _kbhit():  # 若偵測到鍵盤事件
        if ord(msvcrt.getch()) == 32:  # 若鍵盤事件為 空白鍵 = 32
            # press space to end recording
            return True
    return False

def _kbhit():
    ''' Returns True if keyboard character was hit, False otherwise.
    '''

    return msvcrt.kbhit()

def record():
    folder = PATH
    if datasets == 'simulateData':  # Simulate data
        simulate_string = input("Please input the simulate string: ")   # 9_1333 (第幾個_手勢順序)
        folder = f"{folder}/{simulate_string}"
    else:
        folder = f"{PATH}/{GESTURE}"  # Sample data

    if not os.path.exists(folder):
        os.makedirs(folder)  # 產生檔案儲存路徑

    STOP_FLAG = False
    dataInNum = 5   # number of data input
    serialData = serialUSB()  # initializes all required variables, configurations are in config.ini

    try:
        data_total = 0
        while 1:
            data_total += 1
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

            dateTime = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))  # deTime = 當前詳細時間

            sensorFileName = f"{folder}/{GESTURE}_{dateTime}_{USER}" if datasets != 'simulateData' else f"{folder}/{simulate_string}_{USER}"

            with open(f"{sensorFileName}.txt", "w") as sensor_File:
                # 開始紀錄回傳的感測值
                serialData.serialConnection.reset_input_buffer()
                data_buffer = []    #用來暫存每組資料的緩衝區
                first_line = True   #用來標記是否為第一行
                while not _hit_key():
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