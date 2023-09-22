from threading import Thread
import serial
import time
import collections
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import struct
import numpy
import pandas as pd


class serialPlot:
    def __init__(self, serialPort='/dev/ttyACM1', serialBaud=38400, plotLength=100, dataNumBytes=2, dataInNum=1):
        self.port = serialPort
        self.baud = serialBaud
        self.plotMaxLength = plotLength
        self.dataNumBytes = dataNumBytes
        # self.rawData = [bytearray(dataNumBytes) for i in range(dataInNum)]
        self.rawData = [0] * dataInNum
        self.dataInNum = dataInNum
        self.data = [collections.deque([0] * plotLength, maxlen=plotLength) for i in range(dataInNum)]
        self.isRun = True
        self.isReceiving = False
        self.thread = None
        self.plotTimer = 0
        self.previousTimer = 0
        # self.csvData = []

        # self.filter

        print('Trying to connect to: ' + str(serialPort) + ' at ' + str(serialBaud) + ' BAUD.')
        try:
            self.serialConnection = serial.Serial(serialPort, serialBaud, timeout=4)
            print('Connected to ' + str(serialPort) + ' at ' + str(serialBaud) + ' BAUD.')
        except:
            print("Failed to connect with " + str(serialPort) + ' at ' + str(serialBaud) + ' BAUD.')

    def readSerialStart(self):
        if self.thread == None:
            self.thread = Thread(target=self.backgroundThread)
            self.thread.start()
            # Block till we start receiving values
            while self.isReceiving != True:
                print("no data ")
                time.sleep(0.5)

    def getSerialData(self, frame=None, lines=None, lineValueText=None, lineLabel=None, timeText=None, dataInNum=None):
        if frame is not None:
            currentTimer = time.perf_counter()
            self.plotTimer = int((currentTimer - self.previousTimer) * 1000)  # the first reading will be erroneous
            self.previousTimer = currentTimer
            timeText.set_text('Plot Interval = ' + str(self.plotTimer) + 'ms')
            # value,  = struct.unpack('f', self.rawData)    # use 'h' for a 2 byte integer

            value = self.rawData[dataInNum]
            # print(value)
            self.data[dataInNum].append(value)  # we get the latest data point and append it to our array
            lines.set_data(range(self.plotMaxLength), self.data[dataInNum])
            lineValueText.set_text('[' + lineLabel + '] = ' + str(value))
            # self.csvData.append(self.data[-1])
        else:
            for dataInNum in range(self.dataInNum):
                value = self.rawData[dataInNum]
                # print(value)
                self.data[dataInNum].append(value)  # we get the latest data point and append it to our array

    def backgroundThread(self):  # retrieve data
        time.sleep(1.0)  # give some buffer time for retrieving data
        self.serialConnection.reset_input_buffer()
        print("start run")
        while (self.isRun):
            check = self.serialConnection.read().decode("ISO-8859-1")  # Bluetooth 接收與解譯
            print(check)
            if check == 'S':
                for i in range(self.dataInNum):
                    raw = self.serialConnection.read(2)
                    print(raw)
                    self.rawData[i] = int.from_bytes(raw, byteorder='little', signed=True)
                self.isReceiving = True
                print(self.rawData)

    def close(self):
        self.isRun = False
        self.thread.join()
        self.serialConnection.close()
        print('Disconnected...')
        # df = pd.DataFrame(self.csvData)
        # df.to_csv('/home/rikisenia/Desktop/data.csv')


def main():
    # portName = 'COM3'     # for windows users
    portName = '/dev/ttyACM1'
    baudRate = 115200
    maxPlotLength = 100
    dataNumBytes = 4  # number of bytes of 1 data point
    dataInNum = 3
    s = serialPlot(portName, baudRate, maxPlotLength, dataNumBytes, dataInNum)  # initializes all required variables
    try:
        s.readSerialStart()  # starts background thread
    except:
        s.close()


if __name__ == '__main__':
    main()
