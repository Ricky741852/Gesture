/*
  Using the Qwiic Flex Glvoe Controller
  By: Andy England
  SparkFun Electronics
  Date: July 17, 2018
  License: This code is public domain but you buy me a beer if you use this and we meet someday (Beerware license).

  Feel like supporting our work? Buy a board from SparkFun!
  https://www.sparkfun.com/products/14666

  This example shows how to output accelerometer values

  Hardware Connections:
  Attach the Qwiic Shield to your Arduino/Photon/ESP32 or other
  Plug the sensor onto the shield
  Serial.print it out at 115200 baud to serial monitor.
*/

#include "Arduino.h"
#include "ads.h"
#include <SparkFun_ADS1015_Arduino_Library.h>
#include <Wire.h>

#define ADS_RESET_PIN      (3)           // Pin number attached to ads reset line.
#define ADS_INTERRUPT_PIN  (4)           // Pin number attached to the ads data ready line. 

void ads_data_callback(float * sample);
void deadzone_filter(float * sample);
void signal_filter(float * sample);
void parse_com_port(void);



ADS1015 pinkySensor;
ADS1015 indexSensor;
float hand[4] = {0, 0, 0, 0};
//Calibration Array
uint16_t handCalibration[4][2] = {
//{low, hi} switch these to reverse which end is 1 and which is 0  
  {848, 904},  //index
  {851, 908},  //middle
  {718, 940},  //ring
  {761, 912}   //pinky
};

/* Receives new samples from the ADS library */
void ads_data_callback(float * sample, uint8_t sample_type)
{
  if(sample_type == ADS_SAMPLE)
  {
    // Low pass IIR filter
    signal_filter(sample);
  
    // Deadzone filter
    deadzone_filter(sample);
  
    Serial.println(sample[0]);
    
    for (int channel = 0; channel < 2; channel++)
    {
      //Keep in mind that getScaledAnalogData returns a float
      hand[channel] = indexSensor.getScaledAnalogData(channel);
      hand[channel + 2] = pinkySensor.getScaledAnalogData(channel);
    }
    for (int finger = 0; finger < 4; finger++)
    {
      Serial.print(finger);
      Serial.print(": ");
      Serial.print(hand[finger]);
      Serial.print(" ");
    }
    Serial.println();
  }
}

void setup() {
  
  Serial.begin(115200);
  while (!Serial) { delay(10); }
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
  
  /*SparkFun starting setup*/

  // Wire has already begin in ads_init(&init) by Ricky 2023.9.26
  // Wire.begin();
  
  //Begin our finger sensors, change addresses as needed.
  if (pinkySensor.begin(Wire, 100000, ADS1015_ADDRESS_GND) == false) 
  {
     Serial.println("Pinky not found. Check wiring.");
     while (1);
  }
  if (indexSensor.begin(Wire, 100000, ADS1015_ADDRESS_SDA) == false) 
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
  
  // Start reading data in interrupt mode
  ads_run(true);
  /*endregion BendLabs Setup*/
}

void loop() {
  
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
