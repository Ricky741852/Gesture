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
#include <bluefruit.h>
#include <SparkFun_ADS1015_Arduino_Library.h>
#include <Wire.h>

#define ADS_RESET_PIN      (3)           // Pin number attached to ads reset line.
#define ADS_INTERRUPT_PIN  (4)           // Pin number attached to the ads data ready line. 

void ads_data_callback(float * sample);
void deadzone_filter(float * sample);
void signal_filter(float * sample);
void parse_com_port(void);

BLEUart bleuart; // uart over ble

// Define hardware: LED and Button pins and states
const int LED_PIN = 7;
#define LED_OFF LOW
#define LED_ON HIGH

const int BUTTON_PIN = 13;
#define BUTTON_ACTIVE LOW

const int F_AMOUNT = 5; // Number of fingers

union F_DataSplit  // Flex Sensor_Data Split
{
  int full;
  uint8_t byte_data;
} HAND[F_AMOUNT];

ADS1015 pinkySensor;
ADS1015 indexSensor;
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

    // Standardize sample from 0~180 to 100~0
    int stdSample = map(sample[0], 0, 180, 100, 0);
    HAND[0].full = constrain(stdSample, 0, 100);
    
    for (int channel = 0; channel < 2; channel++)
    {
      //Keep in mind that getScaledAnalogData returns a float
      HAND[channel + 1].full = indexSensor.getScaledAnalogData(channel) * 100;
      HAND[channel + 3].full = pinkySensor.getScaledAnalogData(channel) * 100;
    }
    /* Uncomment the following lines to print the data to the Serial Monitor */
    // for (int finger = 0; finger < 5; finger++)
    // {
    //  Serial.print(finger);
    //  Serial.print(": ");
    //  Serial.print(HAND[finger].full);
    //  Serial.print(" ");
    //  Serial.write(HAND[finger].byte_data);
    // }
    // Serial.println();
  }
}

void setup() {
  
  Serial.begin(115200);
  // Uncomment the next line when debugging with the Serial Monitor
//  while (!Serial) { delay(10); }

  #pragma region BendLabs Setup
  
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

  #pragma endregion BendLabs Setup
  
  #pragma region SparkFun Setup

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

  #pragma endregion SparkFun Setup

  #pragma region Bluetooth Setup

  // Initialize Bluetooth:
  pinMode(LED_PIN, OUTPUT); // Turn on-board blue LED off
  digitalWrite(LED_PIN, LED_OFF);
  pinMode(BUTTON_PIN, INPUT);
  
  Bluefruit.begin();
  // Set max power. Accepted values are: -40, -30, -20, -16, -12, -8, -4, 0, 4
  Bluefruit.setTxPower(4);
  Bluefruit.setName("Raytac AT-UART");
  bleuart.begin();

  // Start advertising device and bleuart services
  Bluefruit.Advertising.addFlags(BLE_GAP_ADV_FLAGS_LE_ONLY_GENERAL_DISC_MODE);
  Bluefruit.Advertising.addTxPower();
  Bluefruit.Advertising.addService(bleuart);
  Bluefruit.ScanResponse.addName();

  Bluefruit.Advertising.restartOnDisconnect(true);
  // Set advertising interval (in unit of 0.625ms):
  Bluefruit.Advertising.setInterval(32, 244);
  // number of seconds in fast mode:
  Bluefruit.Advertising.setFastTimeout(30);
  Bluefruit.Advertising.start(0);

  #pragma endregion Bluetooth Setup
  
  // Start reading data in interrupt mode
  ads_run(true);
}

void loop() {
  bleuart.write('e'); // 'e' = 101
  for (int finger = 0; finger < 5; finger++)
  {
    bleuart.write(HAND[finger].byte_data);
  }
  delay(20);

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
