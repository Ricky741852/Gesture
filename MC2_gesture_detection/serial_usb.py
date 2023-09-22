import collections
import serial
import time
from threading import Thread


class serialUSB:
    def __init__(self, serialPort='/dev/ttyACM1', serialBaud=115200, plotLength=100, dataNumBytes=2, dataInNum=3):
        self.serialPort = serialPort
        self.serialBaud = serialBaud
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

        self._connect()

    def _connect(self):
        print('Trying to connect to: ' + str(self.serialPort) + ' at ' + str(self.serialBaud) + ' BAUD.')
        try:
            self.serialConnection = serial.Serial(self.serialPort, self.serialBaud, timeout=4)
            print('Connected to ' + str(self.serialPort) + ' at ' + str(self.serialBaud) + ' BAUD.')
        except:
            print("Failed to connect with " + str(self.serialPort) + ' at ' + str(self.serialBaud) + ' BAUD.')

    def readSerialStart(self):
        if self.thread is None:
            self.thread = Thread(target=self.backgroundThread)
            self.thread.start()
            # Block till we start receiving values
            while not self.isReceiving:
                print("no data ")
                time.sleep(0.5)

    def getSerialData(self):
        return self.data

    def backgroundThread(self):  # retrieve data
        time.sleep(1.0)  # give some buffer time for retrieving data
        self.serialConnection.reset_input_buffer()
        print("start run")
        while self.isRun:
            try:
                check = self.serialConnection.read().decode("ISO-8859-1")  # Bluetooth 接收與解譯
                # print(check)
                if check == 'S':
                    for i in range(self.dataInNum):
                        raw = self.serialConnection.read(2)
                        value = int.from_bytes(raw, byteorder='little', signed=True) * -1
                        self.rawData[i] = value
                        self.data[i].append(value)
                    self.isReceiving = True
                    # print(self.rawData)
            except IOError as exc:
                self._connect()

    def close(self):
        self.isRun = False
        self.thread.join()
        self.serialConnection.close()
        print('Disconnected...')
