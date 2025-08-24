#include <stdint.h>
#include <pico/time.h>
#include <Adafruit_AS5600.h>

// protocol def
const uint8_t PACKET_SIZE = 6;
const uint8_t RESET = 0;
const uint8_t STEP = 1;

// Pin definitions
// motor driver
const int STEP_PIN = D0;
const int DIR_PIN = D1;
static struct repeating_timer timer;
const int MAX_DELAY_US = 100; // toggle every 1ms = 1 rising every 2 ms = 500 steps/s = ~1RPS = slowest
const int MIN_DELAY_US = 15; 

const int MOTOR_ENC = A2; // A2, read with analogWrite until we get AS5600 that has configurable address
const int MOTOR_CPR = 1024;
volatile int MOTOR_OFFSET = 0; // TODO do we need volatile
int MOTOR_LIMIT = 1024/4; // 180 deg of range 
volatile int32_t motorEncoderValue = 0;

volatile unsigned long lastCommandReceived = 0;
const unsigned long COMMAND_TIMEOUT = 500; // ms

Adafruit_AS5600 as5600;


int pModulo(int value, int modulus) {
    return ((value % modulus) + modulus) % modulus;
}


bool timer_callback(struct repeating_timer *t) {
  digitalWrite(STEP_PIN, !digitalRead(STEP_PIN));
  return true; // Keep repeating
}


void setMotorDirection(bool direction) {
  if (direction) {
    digitalWrite(DIR_PIN, LOW);
  } else {
    digitalWrite(DIR_PIN, HIGH);
  }
}


void processMotorCommand(float motor_command, bool direction) {
  setMotorDirection(direction);
      
  if (motor_command == 0) {
    cancel_repeating_timer(&timer); // stop motor
    return;
  }
  
  // map absolute speed (0-1) to timer interval 
  float abs_speed = abs(motor_command);
  int timer_interval = (int)(MAX_DELAY_US - (abs_speed * (MAX_DELAY_US - MIN_DELAY_US)));
  
  cancel_repeating_timer(&timer);
  add_repeating_timer_us(timer_interval, timer_callback, NULL, &timer);
}


void setup() {
  // setup motor pins TODO is this even needed on rp2040?
  pinMode(STEP_PIN, OUTPUT);
  pinMode(DIR_PIN, OUTPUT);
  pinMode(MOTOR_ENC, INPUT);

  MOTOR_OFFSET = analogRead(MOTOR_ENC);

  if (!as5600.begin()) {
    // TODO set user LED to RED
    while (1)
      delay(10);
  }
  as5600.enableWatchdog(false);
  as5600.setPowerMode(AS5600_POWER_MODE_NOM);
  as5600.setHysteresis(AS5600_HYSTERESIS_OFF);
  as5600.setSlowFilter(AS5600_SLOW_FILTER_16X);
  as5600.setFastFilterThresh(AS5600_FAST_FILTER_THRESH_SLOW_ONLY);
  as5600.setZPosition(0);
  as5600.setMPosition(4095);
  as5600.setMaxAngle(4095);

  // TODO set user LED to GREEN

  // setup serial
  Serial.begin(921600); // TODO double check this is needed
}


void loop() {
  motorEncoderValue = analogRead(MOTOR_ENC); // TODO is actually 10 bits, switch to 16bits?
  motorEncoderValue = pModulo(motorEncoderValue - MOTOR_OFFSET, MOTOR_CPR); // [0, CPR]

  if (millis() - lastCommandReceived > COMMAND_TIMEOUT) {
    processMotorCommand(0, true); // kill motor
  }

  // force direction if limit exceeded. don't change commanded speed
  if (motorEncoderValue > MOTOR_LIMIT && motorEncoderValue < (MOTOR_CPR - MOTOR_LIMIT)) {
    if (motorEncoderValue > MOTOR_CPR/2) {
      setMotorDirection(false);
    } else {
      setMotorDirection(true);
    }
  }

  if (motorEncoderValue > MOTOR_LIMIT && motorEncoderValue < (MOTOR_CPR-MOTOR_LIMIT)) {processMotorCommand(0, true);}

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
      // TODO reset encoders?
      processMotorCommand(0, true); // kill motor
      // Discard unnecessary bytes from the serial buffer
      for (int i = 0; i < (PACKET_SIZE - 3); i++) {
        Serial.read();
      }
    }
    else if (command == STEP) {
      // read the two bytes TODO three?
      bool direction = Serial.read();
      uint16_t motor_command;
      Serial.readBytes((char*)&motor_command, sizeof(motor_command));
      motor_command = (float)motor_command / 65535.0;

      processMotorCommand(motor_command, direction);

      int32_t pendulumEncoderValue = as5600.getRawAngle(); // is 12 bits
      unsigned long timestamp = micros();

      Serial.write((uint8_t*)&motorEncoderValue, sizeof(motorEncoderValue));
      Serial.write((uint8_t*)&pendulumEncoderValue, sizeof(pendulumEncoderValue));
      Serial.write((uint8_t*)&timestamp, sizeof(timestamp));
    }
  }
}
