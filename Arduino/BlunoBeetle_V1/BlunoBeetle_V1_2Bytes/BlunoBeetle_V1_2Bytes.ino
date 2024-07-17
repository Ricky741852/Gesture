#include <SparkFun_ADS1015_Arduino_Library.h>
#include "SparkFun_Displacement_Sensor_Arduino_Library.h"
#include "Wire.h"
#include "Timer.h"

Timer trigger;
const int digital_SENSOR_NUM = 5;

int digital_SensorAddr = 10;

union F_DataSplit  // Flex Sensor_Data Split
{
  unsigned char split[2];
  int full;
  byte byte_data[2];
} F_Data[digital_SENSOR_NUM];

ADS digital_FlexSensor;
ADS1015 pinkySensor;
ADS1015 indexSensor;
//Calibration Array
uint16_t handCalibration[4][2] = {
  //{low, hi} switch these to reverse which end is 1 and which is 0
  {797, 893},  //index
  {753, 901},  //middle
  {850, 912},  //ring
  {706, 891}   //pinky
};

void setup() {
  // put your setup code here, to run once:
  Wire.begin();  //啟動I2C協定
  Serial.begin(115200);
  delay(300);
  Serial.println("start");

  if (digital_FlexSensor.begin(digital_SensorAddr) == false) {
    Serial.print("Thumb not found. Check wiring.");
    while (1);
  }

  //Begin our finger sensors, change addresses as needed.
  if (pinkySensor.begin(ADS1015_ADDRESS_VDD) == false)
  {
    Serial.println("Pinky not found. Check wiring.");
    while (1);
  }
  if (indexSensor.begin(ADS1015_ADDRESS_GND) == false)
  {
    Serial.println("Index not found. Check wiring.");
    while (1);
  }

  //Set the calibration values for the hand.
  for (int channel; channel < 2; channel++)
  {
    for (int hiLo = 0; hiLo < 2; hiLo++)
    {
      indexSensor.setCalibration(channel, hiLo, handCalibration[channel][hiLo]);
      pinkySensor.setCalibration(channel, hiLo, handCalibration[channel + 2][hiLo]);
    }
    Serial.println();
  }
  indexSensor.setGain(ADS1015_CONFIG_PGA_TWOTHIRDS);
  pinkySensor.setGain(ADS1015_CONFIG_PGA_TWOTHIRDS);

  //  qwiic_FlexSensor[i].setGain(ADS1015_CONFIG_PGA_TWOTHIRDS);  //Gain of 2/3 to works well with flex glove board voltage swings (default is gain of 2)

  trigger.every(20, Bluetooth_Transfer);  //每20毫秒藍芽傳輸一次，一秒總共傳遞50次數據資料
}

void loop() {
  // put your main code here, to run repeatedly:

  if (digital_FlexSensor.available() != true )
  {
    for (int j = 0; j <= 5; j++)
    {
      digitalWrite(LED_BUILTIN, HIGH);   // turn the LED on (HIGH is the voltage level)
      delay(250);                       // wait for a second
      digitalWrite(LED_BUILTIN, LOW);    // turn the LED off by making the voltage LOW
      delay(250);                       // wait for a second
    }
    delay(500);
  }

  F_Data[0].full = digital_FlexSensor.getX(); //設定上下限

  for (int channel = 0; channel < 2; channel++)
  {
    //Keep in mind that getScaledAnalogData returns a float
    F_Data[channel + 1].full = indexSensor.getAnalogData(channel);
    F_Data[channel + 3].full = pinkySensor.getAnalogData(channel);
  }
//  for (int finger = 0; finger < 5; finger++)
//  {
//    Serial.print(finger);
//    Serial.print(": ");
//    Serial.print(F_Data[finger].full);
//    Serial.print(" ");
//  }
//  Serial.println();
  trigger.update();

}
void Bluetooth_Transfer() {
  Serial.print('e');  //使用e(ascii=101)當作data間隔，每次資料讀取讀1個byte
  for (int i = 0; i < digital_SENSOR_NUM ; i++) {
    Serial.write(F_Data[i].byte_data, 2);
  }
}
