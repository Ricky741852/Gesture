import collections
import serial
import time
from threading import Thread
import configparser

config = configparser.ConfigParser()
config.read('config.ini')
port = config.get('SerialConnection', 'port')
baudrate = config['SerialConnection'].getint('baudrate')
dataSeriesLength = config['SerialConnection'].getint('dataSeriesLength')
# bytesPerDataPoint = config['SerialConnection'].getint('bytesPerDataPoint')
numSensorGroups = config['SerialConnection'].getint('numSensorGroups')

class serialUSB:
    def __init__(self,
                 port=port,
                 baudrate=baudrate,
                 dataSeriesLength=dataSeriesLength,
                 numSensorGroups=numSensorGroups):
        """
        Initializes the serialUSB class.

        Args:
            port (str): The port to connect to.
            baudrate (int): The baudrate for the serial connection.
            dataSeriesLength (int): The length of the data series.
            numSensorGroups (int): The number of sensor groups.

        Returns:
            None
        """
        self.port = port
        self.baudrate = baudrate
        self.dataSeriesLength = dataSeriesLength
        self.numSensorGroups = numSensorGroups
        self.rawData = [0] * numSensorGroups
        self.data = [collections.deque([0] * dataSeriesLength, maxlen=dataSeriesLength) for i in range(numSensorGroups)]
        self.isRun = True
        self.isReceiving = False
        self.thread = None
        self.plotTimer = 0
        self.previousTimer = 0

        self._connect()

    def _connect(self):
        """
        Connects to the specified port at the specified baudrate.

        Args:
            None

        Returns:
            None
        """
        print('Trying to connect to: ' + str(self.port) + ' at ' + str(self.baudrate) + ' BAUD.')
        try:
            self.serialConnection = serial.Serial(self.port, self.baudrate, timeout=4)
            print('Connected to ' + str(self.port) + ' at ' + str(self.baudrate) + ' BAUD.')
        except:
            print("Failed to connect with " + str(self.port) + ' at ' + str(self.baudrate) + ' BAUD.')

    def readSerialStart(self):
        """
        Starts reading serial data in the background.

        Args:
            None

        Returns:
            None
        """
        if self.thread is None:
            self.thread = Thread(target=self.backgroundThread)
            self.thread.start()
            # Block till we start receiving values
            while not self.isReceiving:
                print("no data ")
                time.sleep(0.5)

    
    def getSerialData(self):
        """
        Returns the serial data.
        Used in animation_plot_realtime.py

        Args:
            None

        Returns:
            list: The serial data.
        """
        return self.data

    def backgroundThread(self):
        """
        Background thread for retrieving data from the serial connection.

        Args:
            None

        Returns:
            None
        """
        time.sleep(1.0)  # give some buffer time for retrieving data
        self.serialConnection.reset_input_buffer()
        print("start run")
        while self.isRun:
            try:
                check = self.serialConnection.read().decode("ISO-8859-1")  # Bluetooth 接收與解譯
                # print(check)
                if check == 'S':
                    for i in range(self.numSensorGroups):
                        raw = self.serialConnection.read(2)
                        value = int.from_bytes(raw, byteorder='little', signed=True) * -1
                        self.rawData[i] = value
                        self.data[i].append(value)
                    self.isReceiving = True
                    # print(self.rawData)
            except IOError as exc:
                self._connect()

    def close(self):
        """
        Closes the serial connection.

        Args:
            None

        Returns:
            None
        """
        self.isRun = False
        self.thread.join()
        self.serialConnection.close()
        print('Disconnected...')
