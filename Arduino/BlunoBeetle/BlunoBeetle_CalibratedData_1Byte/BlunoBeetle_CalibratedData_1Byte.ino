#include "Arduino.h"
#include "ads.h"
#include <SparkFun_ADS1015_Arduino_Library.h>
#include "Timer.h"

#define ADS_RESET_PIN      (3)           // Pin number attached to ads reset line.
#define ADS_INTERRUPT_PIN  (5)           // Pin number attached to the ads data ready line. 

void ads_data_callback(float * sample);
void deadzone_filter(float * sample);
void signal_filter(float * sample);

/* Not used in polled mode. Stub function necessary for library compilation */
void ads_data_callback(float * sample, uint8_t sample_type)
{
  
}

Timer trigger;
const int digital_SENSOR_NUM = 5;

union F_DataSplit  // Flex Sensor_Data Split
{
  unsigned char split[2];
  int full;
  byte byte_data;
} F_Data[digital_SENSOR_NUM];

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
  Serial.begin(115200);
  delay(300);
  
  /*BendLabs starting setup*/
  
  Serial.println("Initializing One Axis sensor");

  ads_init_t init;                                // One Axis ADS initialization structure

  init.sps = ADS_50_HZ;                          // Set sample rate to 50 Hz
  init.ads_sample_callback = &ads_data_callback;  // Provide callback for new data
  init.reset_pin = ADS_RESET_PIN;                 // Pin connected to ADS reset line
  init.datardy_pin = ADS_INTERRUPT_PIN;           // Pin connected to ADS data ready interrupt
  init.addr = 10;

  // Initialize ADS hardware abstraction layer, and set the sample rate
  int ret_val = ads_init(&init);

  if(ret_val != ADS_OK)
  {
    Serial.print("One Axis ADS initialization failed with reason: ");
    Serial.println(ret_val);
  }
  else
  {
    Serial.println("One Axis ADS initialization succeeded...");
  }

  //Begin our finger sensors, change addresses as needed.
  if (pinkySensor.begin(Wire, 100000, ADS1015_ADDRESS_VDD) == false) 
  {
     Serial.println("Pinky not found. Check wiring.");
     while (1);
  }
  if (indexSensor.begin(Wire, 100000, ADS1015_ADDRESS_GND) == false) 
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

  ads_run(true);
  // Start reading data in polled mode
  ads_polled(true);

  trigger.every(20, Bluetooth_Transfer);  //每20毫秒藍芽傳輸一次，一秒總共傳遞50次數據資料

  // Wait for first sample
  delay(10);
}

void loop() {

  float sample[2];
  uint8_t data_type;

  // Read data from the one axis ads sensor
  int ret_val = ads_read_polled(sample, &data_type);

  // Check if read was successfull
  if(ret_val == ADS_OK)
  {
    if(data_type == ADS_SAMPLE)
    {
      // Low pass IIR filter
      signal_filter(sample);
    
      // Deadzone filter
      deadzone_filter(sample);

      // Standardize sample from -90~90 to 0~200
      int stdSample = map(sample[0], 250, -175, 0, 200);
      F_Data[0].full = constrain(stdSample, 0, 200); //設定上下限
      
      for (int channel = 0; channel < 2; channel++)
      {        
        F_Data[channel + 1].full = calibration(indexSensor.getAnalogData(channel), channel);
        F_Data[channel + 3].full = calibration(pinkySensor.getAnalogData(channel), channel + 2);
      }
    }
    // Uncommand if you want to show data on the Serial Monitor
//    for (int finger = 0; finger < 5; finger++)
//    {
//      Serial.print(finger);
//      Serial.print(": ");
//      Serial.print(F_Data[finger].full);
//      Serial.print(" ");
//    }
//    Serial.println();  
    
  }
  trigger.update();
}

void Bluetooth_Transfer() {
  Serial.print('e');  //使用e(ascii=101)當作data間隔，每次資料讀取讀兩個bytes
  for (int i = 0; i < digital_SENSOR_NUM ; i++) {
    Serial.write(F_Data[i].byte_data);
  }
}

int calibration(int analogData, int finger)
{
  int low = 0, high = 100;
  int fingerData = map(analogData, handCalibration[finger][0], handCalibration[finger][1], low, high);
  fingerData = ((fingerData) < (low) ? (low) : ((fingerData) > (high) ? (high) : (fingerData)));
//  if((F_Data[finger + 1].full - fingerData) > 25)
//  {
//    return F_Data[finger + 1].full - 10 ;
//  }
  return fingerData;
}

/* 
 *  Second order Infinite impulse response low pass filter. Sample freqency 100 Hz.
 *  Cutoff freqency 20 Hz. 
 */
void signal_filter(float * sample)
{
    static float filter_samples[2][6];

    for(uint8_t i=0; i<2; i++)
    {
      filter_samples[i][5] = filter_samples[i][4];
      filter_samples[i][4] = filter_samples[i][3];
      filter_samples[i][3] = (float)sample[i];
      filter_samples[i][2] = filter_samples[i][1];
      filter_samples[i][1] = filter_samples[i][0];
  
      // 20 Hz cutoff frequency @ 100 Hz Sample Rate
      filter_samples[i][0] = filter_samples[i][1]*(0.36952737735124147f) - 0.19581571265583314f*filter_samples[i][2] + \
        0.20657208382614792f*(filter_samples[i][3] + 2*filter_samples[i][4] + filter_samples[i][5]);   

      sample[i] = filter_samples[i][0];
    }
}

/* 
 *  If the current sample is less that 0.5 degrees different from the previous sample
 *  the function returns the previous sample. Removes jitter from the signal. 
 */
void deadzone_filter(float * sample)
{
  static float prev_sample[2];
  float dead_zone = 0.75f;

  for(uint8_t i=0; i<2; i++)
  {
    if(fabs(sample[i]-prev_sample[i]) > dead_zone)
      prev_sample[i] = sample[i];
    else
      sample[i] = prev_sample[i];
  }
}
