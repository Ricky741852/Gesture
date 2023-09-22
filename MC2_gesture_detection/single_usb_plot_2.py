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
        """serialPlot.__init__

        Args:
            serialPort (str, optional): Serial Port. Defaults to '/dev/ttyACM1'.
            serialBaud (int, optional): Serial Baud Rate. Defaults to 38400.
            plotLength (int, optional): _要繪製幾筆資料. Defaults to 100.
            dataNumBytes (int, optional): Represents the number of bytes per data entry. Defaults to 2.
            dataInNum (int, optional): Amount of fingers. Defaults to 1.
        """
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
        """背景執行緒，用於接收資料
        """
        time.sleep(1.0)  # give some buffer time for retrieving data
        self.reset_input_buffer()
        print("start run")
        while (self.isRun):
            check = self.serialConnection.read().decode("ISO-8859-1")  # Bluetooth 接收與解譯
            # print(check)
            if check == 'S':
                for i in range(self.dataInNum):
                    raw = self.serialConnection.read(2)
                    self.rawData[i] = int.from_bytes(raw, byteorder='little', signed=True) * -1

                self.isReceiving = True
                # print(self.rawData)

    def reset_input_buffer(self):
        """重置連線的input buffer
        """
        self.serialConnection.reset_input_buffer()

    def close(self):
        self.isRun = False
        if self.thread is not None:
            self.thread.join()
        self.serialConnection.close()
        print('Disconnected...')


def main():
    portName = 'COM3'     # for windows users
    # portName = '/dev/ttyACM1'  # for linux users
    # portName = '/dev/ttyUSB0'
    baudRate = 115200
    maxPlotLength = 100
    dataNumBytes = 4  # number of bytes of 1 data point
    dataInNum = 3
    s = serialPlot(portName, baudRate, maxPlotLength, dataNumBytes, dataInNum)  # initializes all required variables
    s.readSerialStart()  # starts background thread

    # plotting starts below
    pltInterval = 20  # Period at which the plot animation updates [ms]
    xmin = 0
    xmax = maxPlotLength
    ymin = -400
    ymax = 400
    fig = plt.figure()
    ax = plt.axes(xlim=(xmin, xmax), ylim=(float(ymin - (ymax - ymin) / 10), float(ymax + (ymax - ymin) / 10)))
    ax.set_title('Arduino Analog Read')
    ax.set_xlabel("time")
    ax.set_ylabel("AnalogRead Value")

    lineLabel = 'I2C 20'
    timeText = ax.text(0.50, 0.95, '', transform=ax.transAxes)
    lines = ax.plot([], [], label=lineLabel)[0]
    lineValueText = ax.text(0.50, 0.90, '', transform=ax.transAxes)
    anim = animation.FuncAnimation(fig, s.getSerialData, fargs=(lines, lineValueText, lineLabel, timeText, 0),
                                   interval=pltInterval)  # fargs has to be a tuple

    lineLabel2 = 'I2C 30'
    timeText2 = ax.text(0.50, 0.85, '', transform=ax.transAxes)
    lines2 = ax.plot([], [], label=lineLabel2)[0]
    lineValueText2 = ax.text(0.50, 0.80, '', transform=ax.transAxes)
    anim2 = animation.FuncAnimation(fig, s.getSerialData, fargs=(lines2, lineValueText2, lineLabel2, timeText2, 1),
                                    interval=pltInterval)

    # print("ploting")
    lineLabel3 = 'I2C 40'
    timeText3 = ax.text(0.50, 0.75, '', transform=ax.transAxes)
    lines3 = ax.plot([], [], label=lineLabel3)[0]
    lineValueText3 = ax.text(0.50, 0.70, '', transform=ax.transAxes)
    anim3 = animation.FuncAnimation(fig, s.getSerialData, fargs=(lines3, lineValueText3, lineLabel3, timeText3, 2),
                                    interval=pltInterval)

    plt.legend(loc="upper left")
    plt.show()

    s.close()


if __name__ == '__main__':
    main()
