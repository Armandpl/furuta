#include <stdint.h>
#include <Encoder.h>

#define PI 3.1415926535897932384626433832795

const float MOTOR_CPR = 400.0;
const float PENDULUM_CPR = 5120.0 * 4;

const float MOTOR_COMMAND_CONVERTER = 65536.0;

// protocol def
const uint8_t PACKET_SIZE = 11;
const uint8_t RESET = 0;
const uint8_t STEP = 1;

// Pin definitions
// motor driver
uint8_t PWM1 = 2;
uint8_t PWM2 = 3;

const uint8_t MOTOR_ENC_A = 9;
const uint8_t MOTOR_ENC_B = 10;

const uint8_t PENDULUM_ENC_A = 7;
const uint8_t PENDULUM_ENC_B = 8;

Encoder motorEncoder(MOTOR_ENC_A, MOTOR_ENC_B);
Encoder pendulumEncoder(PENDULUM_ENC_A, PENDULUM_ENC_B);

volatile unsigned long lastCommandReceived = 0;
const unsigned long COMMAND_TIMEOUT = 500; // ms
float motorDesiredPosition = 0.0;
float motorDesiredVelocity = 0.0;
int32_t motorCommand = 0.0;

float motorPosition = 0.0;
float pendulumPosition =0.0;
float motorVelocity = 0.0;
float pendulumVelocity =0.0;
unsigned long timestamp = 0.0;

float Kp = 3.0 * MOTOR_COMMAND_CONVERTER;
float Kv = 0.05 * MOTOR_COMMAND_CONVERTER;
float TAU = 0.03;


void processMotorCommand(uint16_t motorCommand, bool direction) {
  if (direction) {
    analogWrite(PWM1, motorCommand);
    analogWrite(PWM2, 0);
  }
  else {
    analogWrite(PWM1, 0);
    analogWrite(PWM2, motorCommand);
  }
}


void setup() {
  // setup motor pins
  pinMode(PWM1, OUTPUT);
  pinMode(PWM2, OUTPUT);
  analogWriteResolution(16);
  // setup serial
  Serial.begin(921600);
}


void loop() {
  // check for new message
  if (Serial.available() >= PACKET_SIZE)
  {
    // check for start sequence
    if(Serial.read() != 0x10){
      return;
    }
    if(Serial.read() != 0x02){
      return;
    }
    // valid packet
    lastCommandReceived = millis();

    // read the command type
    uint8_t command = Serial.read();
    if (command == RESET)
    {
      // reset encoders
      motorEncoder.write(0);
      pendulumEncoder.write(0);
      motorDesiredPosition = 0.0;
      motorDesiredVelocity = 0.0;
      motorCommand = 0.0;
      processMotorCommand(0, true); // kill motor
      // Discard unnecessary bytes from the serial buffer
      for (int i = 0; i < (PACKET_SIZE - 3); i++)
      {
        Serial.read();
      }
    }
    else if (command == STEP)
    {
      Serial.readBytes((char*)&motorDesiredPosition, sizeof(motorDesiredPosition));
      Serial.readBytes((char*)&motorDesiredVelocity, sizeof(motorDesiredVelocity));

      Serial.write((uint8_t*)&motorPosition, sizeof(motorPosition));
      Serial.write((uint8_t*)&pendulumPosition, sizeof(pendulumPosition));
      Serial.write((uint8_t*)&motorVelocity, sizeof(motorVelocity));
      Serial.write((uint8_t*)&pendulumVelocity, sizeof(pendulumVelocity));
      Serial.write((uint8_t*)&timestamp, sizeof(timestamp));
      Serial.write((uint8_t*)&motorCommand, sizeof(motorCommand));
    }
  }

  // Watchdog
  if (millis() - lastCommandReceived > COMMAND_TIMEOUT) {
    processMotorCommand(0, true); // kill motor
    return;
  }

  // Read encoders
  float motorPositionNew = motorEncoder.read() * 2 * PI / MOTOR_CPR;
  float pendulumPositionNew = pendulumEncoder.read() * 2 * PI / PENDULUM_CPR;
  unsigned long timestampNew = micros();

  float dt = (timestampNew - timestamp) * 1e-6;

  // Estimate velocity via finite differences and discrete low pass filter
  motorVelocity = (2 * (motorPositionNew - motorPosition) / dt - (1 - 2 * TAU / dt) * motorVelocity) / (1 + 2 * TAU / dt);
  pendulumVelocity = (2 * (pendulumPositionNew - pendulumPosition) / dt - (1 - 2 * TAU / dt) * pendulumVelocity) / (1 + 2 * TAU / dt);

  motorPosition = motorPositionNew;
  pendulumPosition = dt;
  timestamp = timestampNew;

  // Compute motor command with a PID
  motorCommand = Kp * (motorDesiredPosition - motorPosition) + Kv * (motorDesiredVelocity - motorVelocity);

  // process the motor command
  processMotorCommand(abs(motorCommand), (motorCommand < 0.0));
}
