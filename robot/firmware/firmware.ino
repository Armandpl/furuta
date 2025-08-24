#include <stdint.h>
#include "pico/time.h"

// protocol def
const uint8_t PACKET_SIZE = 6;
const uint8_t RESET = 0;
const uint8_t STEP = 1;

// Pin definitions
// motor driver
const int STEP_PIN = D0; // Step pin for stepper motor
const int DIR_PIN = D1;  // Direction pin for stepper motor
static struct repeating_timer timer;
int MAX_DELAY_US = 1000; // toggle very 1ms = 1 rising every 2 ms = 500 steps/s = ~1RPS = slowest
int MIN_DELAY_US = 20; 

const uint8_t MOTOR_ENC = 2; // A2, read with analogWrite until we get AS5600 that has configurable address

volatile unsigned long lastCommandReceived = 0;
const unsigned long COMMAND_TIMEOUT = 500; // ms


void processMotorCommand(uint16_t motor_command, bool direction) {
  if (direction) {
    digitalWrite(DIR_PIN, LOW);
  } else {
    digitalWrite(DIR_PIN, HIGH);
  }
      
  if (speed == 0) {
    cancel_repeating_timer(&timer); // stop motor
    return;
  }
  
  // map absolute speed (0-1) to timer interval 
  float abs_speed = abs(speed);
  int timer_interval = (int)(MAX_DELAY_US - (abs_speed * (MAX_DELAY_US - MIN_DELAY_US)));
  
  cancel_repeating_timer(&timer);
  add_repeating_timer_us(timer_interval, timer_callback, NULL, &timer);
}


void setup() {
  // setup motor pins
  pinMode(STEP_PIN, OUTPUT);
  pinMode(DIR_PIN, OUTPUT);

  // setup serial
  Serial.begin(921600);
}


void loop() {
  if (millis() - lastCommandReceived > COMMAND_TIMEOUT) {
    processMotorCommand(0, true); // kill motor
  }

  if (Serial.available() >= PACKET_SIZE) {
    // check for start sequence
    if(Serial.read() != 0x10){
      return;
    }
    if(Serial.read() != 0x02){
      return;
    }

    lastCommandReceived = millis();

    // valid packet, read the command type
    uint8_t command = Serial.read();
    if (command == RESET) {
      // reset encoders
      motorEncoder.write(0);
      pendulumEncoder.write(0);
      processMotorCommand(0, true); // kill motor
      // Discard unnecessary bytes from the serial buffer
      for (int i = 0; i < (PACKET_SIZE - 3); i++) {
        Serial.read();
      }
    }
    else if (command == STEP) {
      // read the two bytes
      bool direction = Serial.read();
      uint16_t motor_command;
      Serial.readBytes((char*)&motor_command, sizeof(motor_command));

      // process the motor command
      processMotorCommand(motor_command, direction);

      int32_t motorEncoderValue = motorEncoder.read();
      int32_t pendulumEncoderValue = pendulumEncoder.read();
      unsigned long timestamp = micros();

      Serial.write((uint8_t*)&motorEncoderValue, sizeof(motorEncoderValue));
      Serial.write((uint8_t*)&pendulumEncoderValue, sizeof(pendulumEncoderValue));
      Serial.write((uint8_t*)&timestamp, sizeof(timestamp));
    }
  }
}
