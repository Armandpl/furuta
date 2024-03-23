#include <stdint.h>
#include <Encoder.h>

// protocol def
const uint8_t PACKET_SIZE = 6;
const uint8_t RESET = 0;
const uint8_t STEP = 1;

// Pin definitions
const uint8_t MOTOR_DRIVER = 0; // 0 = MC33926
const uint8_t MOTOR_IN1 = 1; // which way to go
const uint8_t MOTOR_IN2 = 2;
const uint8_t MOTOR_D2 = A3; // how fast to go

// uint8_t MOTOR_DRIVER = 1 // 1 = TB9051-FTG
uint8_t PWM1 = 0;
uint8_t PWM2 = 0;

const uint8_t MOTOR_ENC_A = 9;
const uint8_t MOTOR_ENC_B = 10;

const uint8_t PENDULUM_ENC_A = 7;
const uint8_t PENDULUM_ENC_B = 8;

Encoder motorEncoder(MOTOR_ENC_A, MOTOR_ENC_B);
Encoder pendulumEncoder(PENDULUM_ENC_A, PENDULUM_ENC_B);

volatile unsigned long lastCommandReceived = 0;
const unsigned long COMMAND_TIMEOUT = 500; // ms


void processMotorCommand(uint16_t motor_command, bool direction) {
  if (MOTOR_DRIVER == 0) { // MC33926
    if (direction) {
      digitalWrite(MOTOR_IN1, LOW);
      digitalWrite(MOTOR_IN2, HIGH);
    }
    else {
      digitalWrite(MOTOR_IN1, HIGH);
      digitalWrite(MOTOR_IN2, LOW);
    }
    analogWrite(MOTOR_D2, motor_command);
  }
  else if (MOTOR_DRIVER == 1) { // TB9051-FTG
    if (direction) {
      analogWrite(PWM1, motor_command);
      analogWrite(PWM2, 0);
    }
    else {
      analogWrite(PWM1, 0);
      analogWrite(PWM2, motor_command);
    }
  }
}


void setup() {
  // setup motor pins
  if (MOTOR_DRIVER == 0) { // MC33926
    pinMode(MOTOR_IN1, OUTPUT);
    pinMode(MOTOR_IN2, OUTPUT);
    pinMode(MOTOR_D2, OUTPUT);
  }
  else if (MOTOR_DRIVER == 1) { // TB9051-FTG
    pinMode(PWM1, OUTPUT);
    pinMode(PWM2, OUTPUT);
  }
  analogWriteResolution(16);

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
