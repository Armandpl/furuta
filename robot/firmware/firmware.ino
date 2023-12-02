#include <stdint.h>
#include <Encoder.h>

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

const float MOTOR_ENCODER_CPR = 211.2;
const float PENDULUM_ENCODER_CPR = 8192;

Encoder motorEncoder(MOTOR_ENC_A, MOTOR_ENC_B);
Encoder pendulumEncoder(PENDULUM_ENC_A, PENDULUM_ENC_B);


void processMotorCommand(int16_t motor_command) {
  // clamp motor command between -255 and 255
  if (motor_command > 255) {
    motor_command = 255;
  }
  else if (motor_command < -255) {
    motor_command = -255;
  }

  // TODO deadzone
  uint8_t scale_motor_command = 0;
  if (motor_command != 0){
    scale_motor_command = int(0.3 * 255 + 0.7 * abs(motor_command));
  }

  if (MOTOR_DRIVER == 0) { // MC33926
    if (motor_command > 0) {
      digitalWrite(MOTOR_IN1, LOW);
      digitalWrite(MOTOR_IN2, HIGH);
    }
    else {
      digitalWrite(MOTOR_IN1, HIGH);
      digitalWrite(MOTOR_IN2, LOW);
    }
    analogWrite(MOTOR_D2, scale_motor_command);
  }
  else if (MOTOR_DRIVER == 1) { // TB9051-FTG
    if (motor_command > 0) {
      analogWrite(PWM1, scale_motor_command);
      analogWrite(PWM2, 0);
    }
    else {
      analogWrite(PWM1, 0);
      analogWrite(PWM2, scale_motor_command);
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
  analogWriteResolution(8);

  // setup serial
  Serial.begin(57600);
}


void loop() {
  // if at least two bytes (== one motor command)
  // in the serial buffer
  if (Serial.available() >= 2){
    // read the two bytes
    uint8_t low_byte = Serial.read();
    uint8_t high_byte = Serial.read();
    // combine the two bytes to one int16_t
    int16_t motor_command = (high_byte << 8) | low_byte;
    // process the motor command
    processMotorCommand(motor_command);

    // read the motor and pendulum encoder values
    // convert to angles in radians
    // and send them back as two floats

    // read the motor and pendulum encoder values
    long motorEncoderValue = motorEncoder.read();
    long pendulumEncoderValue = pendulumEncoder.read();

    // convert to angles in radians
    float motorAngle = (2 * PI * motorEncoderValue) / MOTOR_ENCODER_CPR;
    float pendulumAngle = (2 * PI * pendulumEncoderValue) / PENDULUM_ENCODER_CPR;

    // send them back as two floats
    Serial.write((uint8_t*)&motorAngle, sizeof(motorAngle));
    Serial.write((uint8_t*)&pendulumAngle, sizeof(pendulumAngle));
  }

}
